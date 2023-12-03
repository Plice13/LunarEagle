import statistics


val_loss_list = [0.39,0.38,0.36]
####----Fit the Model----####
losses = [0.34,0.33,0.31,0.30,0.29,0.27,0.22,0.19,0.16,0.15,0.17,0.17,0.18,0.23,0.30]
x=0
while statistics.mean(val_loss_list[-10:])<statistics.mean(val_loss_list[-11:-2]):
    val_loss_list.append(losses[x])
    x+=1
    print(val_loss_list)