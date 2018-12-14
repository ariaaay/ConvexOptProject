import matplotlib.pyplot as plt
import numpy as np
from util import *
import seaborn as sns

# ksvd = np.loadtxt('./outputs/ksvd_Losses_edit_3.txt',skiprows=1)

transform_algorithm = ['lasso_lars', 'lasso_cd']
fit_algorithm = ['lars','cd']
lambs = np.logspace(-2, 0, 3)
n1 = len(lambs) * len(fit_algorithm) * len(transform_algorithm)
n2 = len(lambs) * len(transform_algorithm)

#real brain + joint
fheight = 15
fwidth = 10


plt.figure(figsize=(fheight,fwidth))
sns.set_palette("hls", n1)
for tr in transform_algorithm:
    for ft in fit_algorithm:
        for lam in lambs:
            try:
                X = np.load('./outputs/brain_joint_loss_X_{}_{}_{}.npy'.format(tr, ft, lam))
                Y = np.load('./outputs/brain_joint_loss_Y_{}_{}_{}.npy'.format(tr, ft, lam))
                loss = X + Y
                ax = sns.lineplot(x=np.arange(len(loss)), y=loss, label='{}_{}(lambda={})'.format(ft, tr, lam))
            except FileNotFoundError:
                continue
for tr in transform_algorithm:
    for lam in lambs:
        try:
            X = np.load('./outputs/brain_joint_loss_X_{}_{}_GD.npy'.format(tr, lam))
            Y = np.load('./outputs/brain_joint_loss_Y_{}_{}_GD.npy'.format(tr, lam))
            loss = X + Y
            ax = sns.lineplot(x=np.arange(len(loss)), y=loss, label='{}_GD(lambda={})'.format(tr, lam))
        except FileNotFoundError:
            continue
plt.yscale("log")
plt.ylabel("Loss")
plt.xlabel("Iterations")
plt.legend()
plt.title("Joint Optimization in Brain and Object Embedding Spaces")
plt.savefig("./figures/joint_brain.png")
# plt.show()

#real brain + alternate
plt.figure(figsize=(fheight,fwidth))
sns.set_palette("hls", n1)
for tr in transform_algorithm:
    for ft in fit_algorithm:
        for lam in lambs:
            try:
                X = np.load('./outputs/brain_loss_X_{}_{}_{}.npy'.format(tr, ft, lam))
                Y = np.load('./outputs/brain_loss_Y_{}_{}_{}.npy'.format(tr, ft, lam))
                loss = X + Y
                ax = sns.lineplot(x=np.arange(len(loss)), y=loss, label='{}_{}(lambda={})'.format(ft, tr, lam))
            except FileNotFoundError:
                continue
plt.yscale("log")
plt.ylabel("Loss")
plt.xlabel("Iterations")
plt.legend()
plt.title("Alternate Optimization in Brain and Object Embedding Spaces")
plt.savefig("./figures/alternate_brain.png")
# plt.show()

#simulation + joint
plt.figure(figsize=(fheight,fwidth))
sns.set_palette("hls", n1+n2)
for tr in transform_algorithm:
    for ft in fit_algorithm:
        for lam in lambs:
            try:
                X = np.load('./outputs/sim_joint_loss_X_{}_{}_{}.npy'.format(tr, ft, lam))
                Y = np.load('./outputs/sim_joint_loss_Y_{}_{}_{}.npy'.format(tr, ft, lam))
                loss = X + Y
                ax = sns.lineplot(x=np.arange(len(loss)), y=loss, label='{}_{}(lambda={})'.format(ft, tr, lam))
            except FileNotFoundError:
                continue

for tr in transform_algorithm:
    for lam in lambs:
        try:
            X = np.load('./outputs/sim_joint_loss_X_{}_{}_GD.npy'.format(tr, lam))
            Y = np.load('./outputs/sim_joint_loss_Y_{}_{}_GD.npy'.format(tr, lam))
            loss = X + Y
            ax = sns.lineplot(x=np.arange(len(loss)), y=loss, label='{}_GD(lambda={})'.format(tr, lam))
        except FileNotFoundError:
            continue
plt.yscale("log")
plt.ylabel("Loss")
plt.xlabel("Iterations")
plt.legend()
plt.title("Joint Optimization in Simulated Data")
plt.savefig("./figures/joint_simulation.png")
# plt.show()


#simulation + alternate
plt.figure(figsize=(fheight,fwidth))
sns.set_palette("hls", n1)
for tr in transform_algorithm:
    for ft in fit_algorithm:
        for lam in lambs:
            try:
                X = np.load('./outputs/loss_X_{}_{}_{}.npy'.format(tr, ft, lam))
                Y = np.load('./outputs/loss_Y_{}_{}_{}.npy'.format(tr, ft, lam))
                loss = X + Y
                ax = sns.lineplot(x=np.arange(len(loss)), y=loss, label='{}_{}(lambda={})'.format(ft, tr, lam))
            except FileNotFoundError:
                continue
plt.yscale("log")
plt.ylabel("Loss")
plt.xlabel("Iterations")
plt.legend()
plt.title("Alternate Optimization in Simulated Data")
plt.savefig("./figures/alternate_simulation")
# plt.show()

# #brain + joint GD
# plt.figure(figsize=(fheight,fwidth))
# sns.set_palette("hls", n2)
# for tr in transform_algorithm:
#     for lam in lambs:
#         try:
#             X = np.load('./outputs/brain_joint_loss_X_{}_{}_GD.npy'.format(tr, lam))
#             Y = np.load('./outputs/brain_joint_loss_Y_{}_{}_GD.npy'.format(tr, lam))
#             loss = X + Y
#             ax = sns.lineplot(x=np.arange(len(loss)), y=loss, label='{}_GD(lambda={})'.format(tr, lam))
#         except FileNotFoundError:
#             continue
# plt.yscale("log")
# plt.ylabel("Loss")
# plt.xlabel("Iterations")
# plt.legend()
# plt.title("Joint Optimization in Brain and Object Embedding Spaces (with gradient descent)")
# plt.savefig("./figures/joint_gd_brain.png")
# # plt.show()
#
# #brain + joint GD
# plt.figure(figsize=(fheight,fwidth))
# sns.set_palette("hls", n2)
# for tr in transform_algorithm:
#     for lam in lambs:
#         try:
#             X = np.load('./outputs/sim_joint_loss_X_{}_{}_GD.npy'.format(tr, lam))
#             Y = np.load('./outputs/sim_joint_loss_Y_{}_{}_GD.npy'.format(tr, lam))
#             loss = X + Y
#             ax = sns.lineplot(x=np.arange(len(loss)), y=loss, label='{}(lambda={})'.format(tr, lam))
#         except FileNotFoundError:
#             continue
# plt.yscale("log")
# plt.ylabel("Loss")
# plt.xlabel("Iterations")
# plt.legend()
# plt.title("Joint Optimization in Simulated Data (with gradient descent)")
# plt.savefig("./figures/joint_gd_simulation")
# # plt.show()