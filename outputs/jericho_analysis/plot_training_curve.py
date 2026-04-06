import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Run 0: ppo_epochs=1, n=8, lr=1e-6, bs=1
run0_val = [
    10, 10, 25, 10, 30, 10, 15, 10, 20, 10, 35, 20, 15, 20, 25, 10, 20, 10,
    30, 20, 15, 10, 20, 10, 35, 10, 10, 10, 25, 10, 25, 10, 30, 20, 10, 10,
    20, 20, 35, 10, 0, 10, 25, 0, 0, 10, 30, 10, 15, 10
]

# Run 1: ppo_epochs=2, n=8, lr=1e-6, bs=1
run1_val = [
    5.0, 5.0, 15.0, 15.0, 40.0, 15.0, 5.0, 5.0, 15.0, 5.0,
    15.0, 5.0, 5.0, 15.0, 15.0, 5.0, 5.0, 5.0, 5.0, 15.0,
    15.0, 5.0, 5.0, 15.0, 15.0, 5.0, 5.0, 40.0, 5.0, 5.0,
    15.0, 5.0, 5.0, 5.0, 5.0, 15.0, 5.0, 40.0, 5.0, 5.0,
    5.0, 15.0, 15.0, 15.0, 5.0, 5.0, 5.0, 5.0, 0.0, 15.0,
]

# Run 2: ppo_epochs=1, n=16, lr=1e-6, bs=1
run2_val = [
    15.0, 15.0, 40.0, 15.0, 5.0, 15.0, 40.0, 5.0, 15.0, 15.0,
    5.0, 5.0, 5.0, 5.0, 5.0, 15.0, 30.0, 5.0, 15.0, 15.0,
    15.0, 15.0, 5.0, 5.0, 15.0, 30.0, 5.0, 15.0, 15.0, 0.0,
    5.0, 15.0, 5.0, 5.0, 5.0, 15.0, 5.0, 40.0, 40.0, 5.0,
    5.0, 15.0, 15.0, 5.0, 15.0, 15.0, 5.0, 40.0, 15.0, 5.0,
]

# Run 3: ppo_epochs=1, n=32, lr=1e-6, bs=1 (recovered from tensorboard)
run3_val = [
    5.0, 15.0, 30.0, 15.0, 5.0, 5.0, 15.0, 5.0, 15.0, 15.0,
    15.0, 15.0, 15.0, 15.0, 5.0, 5.0, 5.0, 15.0, 15.0, 15.0,
    5.0, 15.0, 15.0, 5.0, 5.0, 15.0, 0.0, 5.0, 15.0, 15.0,
    15.0, 40.0, -5.0, 15.0, 15.0, 35.0, 5.0, 40.0, 5.0, 5.0,
    5.0, 5.0, 15.0, 5.0, 5.0, 15.0, 5.0, 15.0, 5.0, 15.0,
]

# Run 4: ppo_epochs=4, n=8, lr=1e-6, bs=1
run4_val = [
    15.0, 5.0, 15.0, 15.0, 15.0, 40.0, 15.0, 5.0, 15.0, 15.0,
    40.0, 30.0, 40.0, 5.0, 15.0, 15.0, 5.0, 15.0, 15.0, 30.0,
    15.0, 40.0, 15.0, 40.0, 5.0, 30.0, 15.0, 15.0, 5.0, 40.0,
    5.0, 15.0, 5.0, 15.0, 15.0, 30.0, 15.0, 15.0, 5.0, 5.0,
    5.0, 15.0, 35.0, 5.0, 15.0, 5.0, 15.0, 15.0, 5.0, 30.0,
]

# Run 5: ppo_epochs=4, n=8, lr=1e-5, bs=1
run5_val = [
    15.0, 5.0, 30.0, 15.0, -5.0, 15.0, 15.0, 5.0, 5.0, 15.0,
    5.0, 5.0, 15.0, 15.0, 15.0, 30.0, 15.0, 30.0, 25.0, 35.0,
    30.0, 5.0, 5.0, 25.0, 15.0, 40.0, 45.0, 35.0, 40.0, 35.0,
    39.0, 30.0, 40.0, 45.0, 39.0, 40.0, 29.0, 35.0, 30.0, 35.0,
    35.0, 15.0, 35.0, 35.0, 40.0, 35.0, 30.0, 35.0, 40.0, 35.0,
]

# Run 6: ppo_epochs=1, n=8, lr=1e-5, bs=8
run6_val = [
    15.0, 30.0, 5.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0,
    5.0, 15.0, 40.0, 30.0, 30.0, 45.0, 30.0, 35.0, 30.0, 40.0,
    40.0, 40.0, 30.0, 40.0, 45.0, 45.0, 15.0, 40.0, 55.0, 45.0,
    55.0, 55.0, 45.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0,
    55.0, 45.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0,
]


def _mean(vals):
    return sum(vals) / len(vals)


def plot_by_n(output_path):
    """Compare different n values (fixed ppo_epochs=1, lr=1e-6, bs=1)."""
    steps = list(range(1, 51))
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(steps, run0_val, 'g-o', markersize=4, label=f'n=8  (mean={_mean(run0_val):.1f})')
    ax.plot(steps, run2_val, 'b-s', markersize=4, label=f'n=16 (mean={_mean(run2_val):.1f})')
    ax.plot(steps, run3_val, 'm-^', markersize=4, label=f'n=32 (mean={_mean(run3_val):.1f})')

    ax.set_xlabel('Step (Epoch)', fontsize=12)
    ax.set_ylabel('Zork1 Game Score', fontsize=12)
    ax.set_title('Effect of Rollout Count (n) — fixed ppo_epochs=1, lr=1e-6, bs=1', fontsize=14)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'Saved to {output_path}')


def plot_by_ppo_epochs(output_path):
    """Compare different ppo_epochs (fixed n=8, lr=1e-6, bs=1)."""
    steps = list(range(1, 51))
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(steps, run0_val, 'g-o', markersize=4, label=f'ppo_epochs=1 (mean={_mean(run0_val):.1f})')
    ax.plot(steps, run1_val, 'r-s', markersize=4, label=f'ppo_epochs=2 (mean={_mean(run1_val):.1f})')
    ax.plot(steps, run4_val, 'b-^', markersize=4, label=f'ppo_epochs=4 (mean={_mean(run4_val):.1f})')

    ax.set_xlabel('Step (Epoch)', fontsize=12)
    ax.set_ylabel('Zork1 Game Score', fontsize=12)
    ax.set_title('Effect of PPO Epochs — fixed n=8, lr=1e-6, bs=1', fontsize=14)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'Saved to {output_path}')


def plot_by_lr(output_path):
    """Compare different learning rates (fixed n=8, ppo_epochs=4, bs=1)."""
    steps = list(range(1, 51))
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(steps, run4_val, 'b-o', markersize=4, label=f'lr=1e-6 (mean={_mean(run4_val):.1f})')
    ax.plot(steps, run5_val, 'r-s', markersize=4, label=f'lr=1e-5 (mean={_mean(run5_val):.1f})')

    ax.set_xlabel('Step (Epoch)', fontsize=12)
    ax.set_ylabel('Zork1 Game Score', fontsize=12)
    ax.set_title('Effect of Learning Rate — fixed n=8, ppo_epochs=4, bs=1', fontsize=14)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'Saved to {output_path}')


def plot_by_bs(output_path):
    """Compare batch size=1 vs 8 (both with n=8, lr=1e-5)."""
    steps = list(range(1, 51))
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(steps, run5_val, 'b-o', markersize=4, label=f'bs=1, ppo=4 (mean={_mean(run5_val):.1f})')
    ax.plot(steps, run6_val, 'r-s', markersize=4, label=f'bs=8, ppo=1 (mean={_mean(run6_val):.1f})')

    ax.axhline(y=42.1, color='k', linestyle='--', alpha=0.5, label='JitRL mean (42.1)')

    ax.set_xlabel('Step (Epoch)', fontsize=12)
    ax.set_ylabel('Zork1 Game Score', fontsize=12)
    ax.set_title('Effect of Batch Size — fixed n=8, lr=1e-5', fontsize=14)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'Saved to {output_path}')


def plot_all(output_path):
    """All runs in one plot."""
    fig, ax = plt.subplots(figsize=(14, 7))

    runs = [
        (run0_val, 'g-o', 'n=8, ppo=1, lr=1e-6, bs=1'),
        (run1_val, 'r-s', 'n=8, ppo=2, lr=1e-6, bs=1'),
        (run4_val, 'b-^', 'n=8, ppo=4, lr=1e-6, bs=1'),
        (run2_val, 'c-D', 'n=16, ppo=1, lr=1e-6, bs=1'),
        (run3_val, 'm-v', 'n=32, ppo=1, lr=1e-6, bs=1'),
        (run5_val, 'y-*', 'n=8, ppo=4, lr=1e-5, bs=1'),
        (run6_val, 'k-P', 'n=8, ppo=1, lr=1e-5, bs=8'),
    ]
    for data, style, name in runs:
        steps = list(range(1, len(data) + 1))
        ax.plot(steps, data, style, markersize=3, label=f'{name} (mean={_mean(data):.1f})', alpha=0.7)

    ax.axhline(y=42.1, color='k', linestyle='--', alpha=0.5, label='JitRL mean (42.1)')

    ax.set_xlabel('Step (Epoch)', fontsize=12)
    ax.set_ylabel('Zork1 Game Score', fontsize=12)
    ax.set_title('GRPO Training on Zork1 - All Experiments', fontsize=14)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'Saved to {output_path}')


if __name__ == '__main__':
    plot_by_n('outputs/jericho_analysis/val_score_by_n.png')
    plot_by_ppo_epochs('outputs/jericho_analysis/val_score_by_ppo_epochs.png')
    plot_by_lr('outputs/jericho_analysis/val_score_by_lr.png')
    plot_by_bs('outputs/jericho_analysis/val_score_by_bs.png')
    plot_all('outputs/jericho_analysis/val_score_all.png')
