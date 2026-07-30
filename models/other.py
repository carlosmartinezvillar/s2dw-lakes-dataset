import torch
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts, SequentialLR
import models

def plot_scheduler():
	dummy = torch.nn.Parameter(torch.zeros(1))
	optimizer = torch.optim.AdamW([dummy], lr=0.005)

	epochs = 95
	warmup_epochs = 5
	cosine_epochs = epochs - warmup_epochs
	cycles = 2
	cosine_steps  = cosine_epochs // cycles
	eta_min = 0.0

	warmup = LinearLR(
		optimizer, start_factor=1e-2, total_iters=warmup_epochs)
	cosine = CosineAnnealingWarmRestarts(
		optimizer, T_0=cosine_steps, T_mult=1, eta_min=eta_min)
	scheduler = SequentialLR(
		optimizer,
		schedulers=[warmup,cosine],
		milestones=[warmup_epochs]
	)

	lrs = []
	for epoch in range(epochs):
		lrs.append(optimizer.param_groups[0]['lr'])
		scheduler.step()


	fig,ax = plt.subplots()
	ax.plot(range(epochs), lrs, linewidth=1.5)

	ax.set_xlabel('Epoch')
	ax.set_ylabel('Learning rate')
	ax.set_title('Warmup + Cosine Annealing Warm Restarts')
	ax.legend()
	fig.tight_layout()
	plt.show()


if __name__ == '__main__':
	plot_scheduler()