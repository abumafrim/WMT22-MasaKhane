import matplotlib.pyplot as plt
import os



FOLDER = "/home/mila/c/chris.emezue/wmt22/unfiltered"
LOG_FILENAME = 'train_unfiltered.log'

log_file = os.path.join(FOLDER,LOG_FILENAME)


with open(log_file,'r') as file:
    lines = file.readlines() 
lines_ = [l for l in lines if l.startswith('Epoch')]
losses = [float(l.split('Avg. loss:')[1].strip()) for l in lines_]
x_ = [i for i in range(len(losses))]

fig,ax = plt.subplots()

ax.plot(x_,losses)
ax.set_xlabel('Train step')
ax.set_ylabel('Train Loss')
ax.set_title("Training loss of MMTAfrica Large finetuning on WMT22")

fig.savefig('./train_losses.png')