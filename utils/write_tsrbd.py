def write_tensorboard(loss_dict, writer, epoch, lr, t, flag):
        # for loss in loss_dict.keys():
        #     writer.add_scalar(f"{loss}_{flag}", np.mean(loss_dict[loss]), epoch)
        # writer.add_scalar("mse_train", np.mean(mse_train), epoch)

        # writer.add_scalar("learning_rate", np.mean(lr), epoch)
        # writer.add_scalar("learning_rate", np.mean(optimizer.param_groups[0]["lr"]), epoch)

        # writer.add_scalar("time", np.mean(time.time() - t), epoch)
        
        raise NotImplementedError