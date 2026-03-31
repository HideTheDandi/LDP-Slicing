import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR
from ldp_slicing import WC_ALIASES, dp_slicing_dwt, get_privacy_budget
from experiment.models.resnet_cifar import resnet56
# from dp_slicing_batch import dp_slicing_perceptual 

#Spaghetti code for training privacy-preserving image classification ResNet-56 with LDP-Slicing

# --- CIFAR Normalization Constants ---
# Using CIFAR-10 stats for both datasets 
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]

# ---------------------- Data Loading -------------------------------------------------------



def build_data_loaders(dataset, batch_size, num_workers=4, pin_memory=True, cutout=False):
    assert dataset in ["cifar10", "cifar100"]
    dataset_class = torchvision.datasets.CIFAR10 if dataset == "cifar10" else torchvision.datasets.CIFAR100
    
    # Training transforms (data augmentation)
    train_transforms_list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    
    # Add Cutout
    if cutout:
        from experiment.utils import Cutout
        train_transforms_list.append(Cutout(n_holes=1, length=16))
    
    train_transform = transforms.Compose(train_transforms_list)
    
    # Test transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Load datasets
    train_dataset = dataset_class(
        root='./data', train=True, download=True, transform=train_transform)
    test_dataset = dataset_class(
        root='./data', train=False, download=True, transform=test_transform)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True)  # drop_last for stable BN
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory)
    
    return train_loader, test_loader, len(train_dataset), len(test_dataset)

# ---------------------- Privacy Transform -------------------------------------------------

def apply_privacy_transform(x, args, epsilon_y, epsilon_c):
    # Args:
    #     x: Input tensor [B, C, H, W] in range [0, 1]
    #     args: Arguments containing dp method and configuration
    #     epsilon_y: Privacy budget for Y channel
    #     epsilon_c: Privacy budget for Cb/Cr channels
    
    # Returns:
    #     Tensor [B, C, H, W] in range [0, 1]
    #
    if args.dp_method == "none":
        return x
    
    with torch.no_grad():
        # if args.dp_method == "dct":
        #     if dp_slicing_perceptual is None:
        #         raise RuntimeError(
        #             "DCT error"
        #         )
        #     protected = dp_slicing_perceptual(
        #         x,
        #         chs_remove=[0],  # Remove DC coefficient
        #         epsilon_y=epsilon_y,
        #         epsilon_c=epsilon_c,
        #         device=x.device
        #     )
        if args.dp_method == "dwt":
            # DWT-based DP 
            protected = dp_slicing_dwt(
                x,
                wavelet=args.wavelet,
                level=1,  # Fixed to level 1 as per requirement
                remove_ll=True,  # Remove LL sub-band
                epsilon_y=epsilon_y,
                epsilon_c=epsilon_c,
                device=x.device
            )
        else:
            raise ValueError(f"Unknown dp_method: {args.dp_method}")
        
        # Safety: clamp to valid range before normalization
        protected = protected.clamp(0, 1)
        return protected

# ------------------------------ Training ----------------------------------------------

def train_epoch(net, train_loader, criterion, optimizer, epoch, args, normalize, epsilon_y, epsilon_c):
    """Train for one epoch"""
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        
        # Apply privacy transform (DCT stays on GPU, DWT may have CPU overhead)
        if args.dp_method != "none":
            inputs = apply_privacy_transform(inputs, args, epsilon_y, epsilon_c)
        
        # 2) Normalize after privacy transform
        inputs = normalize(inputs).contiguous()
        
        # 3) Forward and backward
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # 4) Gradient clipping (only for DP methods)
        if args.dp_method != "none" and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.grad_clip)
        
        optimizer.step()
        
        # 5) Track metrics
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # 6) Log progress
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
            print(f'  Step [{batch_idx+1}/{len(train_loader)}]: '
                  f'Loss={train_loss/(batch_idx+1):.3f}, '
                  f'Acc={100.*correct/total:.3f}% ({correct}/{total})')
    
    # Epoch summary
    epoch_loss = train_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    print(f'\nEpoch {epoch} Summary:')
    print(f'  Train Loss: {epoch_loss:.4f}')
    print(f'  Train Acc: {epoch_acc:.2f}%')
    
    return epoch_loss, epoch_acc

def test_epoch(net, test_loader, criterion, epoch, args, normalize, epsilon_y, epsilon_c):
    """Test for one epoch"""
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            
            # 1) Apply privacy transform (same as training)
            if args.dp_method != "none":
                inputs = apply_privacy_transform(inputs, args, epsilon_y, epsilon_c)
            
            # 2) Normalize
            inputs = normalize(inputs).contiguous()
            
            # 3) Forward
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            
            # 4) Track metrics
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    # Test summary
    test_loss = test_loss / len(test_loader)
    test_acc = 100. * correct / total
    
    print(f'\n{"="*80}')
    print(f'TEST RESULTS - Epoch {epoch}')
    print(f'{"="*80}')
    print(f'Loss: {test_loss:.4f} | Accuracy: {test_acc:.2f}% ({correct}/{total})')
    if args.dp_method != "none":
        print(f'(Tested on {args.dp_method.upper()}-DP protected images)')
    print(f'{"="*80}')
    
    return test_loss, test_acc

# ------------------------- Main --------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='ResNet-56 CIFAR Training with DP-Slicing')

    parser.add_argument('--dataset', type=str, default='cifar10', 
                       choices=['cifar10', 'cifar100'], help='Dataset name')
    parser.add_argument('--batchsize', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=250, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--milestones', type=int, nargs='+', default=[80, 160, 200],
                       help='Learning rate decay milestones')
    parser.add_argument('--gamma', type=float, default=0.5, help='LR decay factor')
    parser.add_argument('--grad_clip', type=float, default=0.5, 
                       help='Gradient clipping max norm (0 to disable, only used with DP)')
    
    # Privacy arguments
    parser.add_argument(
        '--dp_method',
        type=str,
        default='dwt',
        choices=['none', 'dct', 'dwt'],
        help='Privacy method (dct is optional and may be unavailable).',
    )
    parser.add_argument('--epsilon', type=float, default=5.2,
                       choices=[1.0, 2.4, 5.2, 12.0, 20.0, 32.0, 58.0],
                       help='Total privacy budget (ε_tot)')
    parser.add_argument(
        '--color_weight',
        type=str,
        default='411',
        choices=tuple(WC_ALIASES.keys()),
        help='Color-weight ratio alias. 411->4:1:1, 211->2:1:1, 111->1:1:1.',
    )
    parser.add_argument('--eval_only', action='store_true', help='Only run evaluation on the test set and exit (requires --resume).')
    
    # DWT options
    parser.add_argument('--wavelet', type=str, default='haar',
                       choices=['haar', 'db2', 'sym2', 'coif1', 'bior1.3', 'rbio1.3'],
                       help='Wavelet type for DWT (only used if dp_method=dwt)')
    
    # Cutout
    parser.add_argument('--cutout', action='store_true', help='Use Cutout augmentation')
    
    # Misc 
    parser.add_argument('--sess', type=str, default=None, help='Session name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--deterministic', action='store_true', 
                       help='Use deterministic algorithms (slower but reproducible)')
    parser.add_argument('--resume', type=str, default=None, 
                       help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Session name
    if args.sess is None:
        if args.dp_method == "none":
            privacy_str = "baseline"
        elif args.dp_method == "dct":
            privacy_str = f"dct_dc0_eps{args.epsilon}"
        else:  # dwt
            privacy_str = f"dwt_{args.wavelet}_LL1_eps{args.epsilon}"
        args.sess = f"resnet56_{args.dataset}_{privacy_str}_new"
    
    # Random seeds
    assert torch.cuda.is_available(), 'CUDA is required for this script.'
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        print("⚠️  Running in deterministic mode (slower)")
    else:
        cudnn.benchmark = True
    
    # Get privacy budget configuration
    if args.dp_method != "none":
        epsilon_y, epsilon_c, total_eps, budget_key = get_privacy_budget(
            color_weight=args.color_weight,
            total_epsilon=args.epsilon,
        )
    else:
        epsilon_y = None
        epsilon_c = None
        total_eps = 0.0
    
    
    # Dataloaders
    print('\n==> Preparing data..')
    train_loader, test_loader, n_train, n_test = build_data_loaders(
        args.dataset, args.batchsize, num_workers=args.num_workers, cutout=args.cutout)
    print(f"Training samples: {n_train}")
    print(f"Test samples: {n_test}")
    
    normalize = transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)
    num_classes = 10 if args.dataset == "cifar10" else 100
    print(f"Using CIFAR normalization: mean={CIFAR_MEAN}, std={CIFAR_STD}")
        # Build model
    print('\n==> Building model..')
    net = resnet56(num_classes=num_classes)
    net = net.cuda()
    net = torch.nn.DataParallel(net)
    print(f"Model: ResNet-56 with {num_classes} classes")
    print(f"Total parameters: {sum(p.numel() for p in net.parameters()):,}")
    
    # Otimizer and loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    scheduler = MultiStepLR(
        optimizer,
        milestones=args.milestones,
        gamma=args.gamma
    )
    
    print(f"\nOptimizer: SGD (lr={args.lr}, momentum={args.momentum}, wd={args.weight_decay})")
    print(f"Scheduler: MultiStepLR (milestones={args.milestones}, gamma={args.gamma})")
    
    # Resume from checkpoint 
    start_epoch = 1
    best_acc = 0.0
    if args.resume:
        print(f'\n==> Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume)
        net.load_state_dict(checkpoint['net'])
        if not args.eval_only:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['acc']
            print(f'Resumed from epoch {checkpoint["epoch"]}, best acc: {best_acc:.2f}%')
        else:
            print('Loaded checkpoint for eval_only.')

    if args.eval_only:
        if not args.resume:
            raise ValueError('--eval_only requires --resume <checkpoint>')
        test_loss, test_acc = test_epoch(
            net, test_loader, criterion, epoch=0, args=args,
            normalize=normalize, epsilon_y=epsilon_y, epsilon_c=epsilon_c
        )
        print(f'\nEVAL ONLY RESULT: Loss={test_loss:.4f} | Accuracy={test_acc:.2f}%')
        return
    
    # Training loop
    os.makedirs('./checkpoint', exist_ok=True)
    
    print('\n' + '='*80)
    print('STARTING TRAINING')
    print('='*80)
    
    for epoch in range(start_epoch, args.epochs + 1):
        # Training
        print(f'\n{"="*80}')
        print(f'Privacy budget ε_tot: {total_eps:.4f}' if args.dp_method != "none" else 'No Privacy Protection (Baseline)')
        print(f'Epoch: {epoch}/{args.epochs}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'{"="*80}')
        
        train_loss, train_acc = train_epoch(
            net, train_loader, criterion, optimizer, epoch, args, 
            normalize, epsilon_y, epsilon_c)
        
        # Testing
        test_loss, test_acc = test_epoch(
            net, test_loader, criterion, epoch, args,
            normalize, epsilon_y, epsilon_c)
        
        # Save checkpoint if best
        if test_acc > best_acc:
            print(f'✓ New best accuracy! Saving checkpoint... (Previous: {best_acc:.2f}%)')
            best_acc = test_acc
            
            state = {
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'acc': test_acc,
                'epoch': epoch,
                'args': vars(args),
                'epsilon_y': epsilon_y,
                'epsilon_c': epsilon_c,
            }
            
            torch.save(state, f'./checkpoint/{args.sess}_best.pth')
            print(f'Checkpoint saved to: ./checkpoint/{args.sess}_best.pth')
        
        # Save latest checkpoint every 10 epochs
        if epoch % 10 == 0:
            state = {
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'acc': test_acc,
                'epoch': epoch,
                'args': vars(args),
                'epsilon_y': epsilon_y,
                'epsilon_c': epsilon_c,
            }
            torch.save(state, f'./checkpoint/{args.sess}_latest.pth')
        
        # Log to file
        log_file = f'./checkpoint/{args.sess}_log.txt'
        with open(log_file, 'a') as f:
            f.write(f'Epoch {epoch}: LR={optimizer.param_groups[0]["lr"]:.6f}, '
                   f'Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, '
                   f'Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%, '
                   f'Best={best_acc:.2f}%\n')
        
        # Step scheduler
        scheduler.step()
    
    # Training completed
    print('\n' + '='*80)
    print('TRAINING COMPLETED')
    print('='*80)
    print(f'Best test accuracy: {best_acc:.2f}%')
    print(f'Checkpoint saved to: ./checkpoint/{args.sess}_best.pth')
    if args.dp_method != "none":
        if args.dp_method == "dct":
            print(f'Privacy Method: DCT (DC removed)')
        else:
            print(f'Privacy Method: DWT ({args.wavelet}, LL1 removed)')
        print(f'Total Privacy Budget: ε_tot = {total_eps:.4f}')
    print('='*80)

if __name__ == '__main__':
    main()
