"""General-purpose training script for image-to-image translation.

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train img2img 3D pose model:
     python train.py --use_aug --gpu_ids 0,1,2,3 --name train_experiment_name --model cycle_3Dpose_gan

See options/base_options.py and options/train_options.py for more training options.
"""

import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def init_process(rank, size, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '28500'
    dist.init_process_group(backend, rank=rank, world_size=size)
#
def train_main(rank, world_size, opt):

    init_process(rank, world_size)
    torch.cuda.set_device(torch.device('cuda:{}'.format(rank)))

    dataset = create_dataset(opt, rank)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    if opt.max_dataset_size == float("inf"):
        opt.max_dataset_size =  int(dataset_size /opt.batch_size) * opt.batch_size

    model = create_model(opt, rank)      # create a model given opt.model and other options

    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations

    if rank == 0:
        visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        if rank == 0:
            visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if rank == 0:
                # display images on visdom and save images to a HTML file
                if total_iters % opt.display_freq == 0:
                    save_result = total_iters % opt.update_html_freq == 0
                    model.compute_visuals()
                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                loss_log_data = {"Epoch": epoch}
                for loss_name, loss_value in losses.items():
                    loss_log_data[loss_name] = loss_value

                if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                    print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                    save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                    model.save_networks(save_suffix)


            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size

                if rank == 0:
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                    if opt.display_id > 0:
                        visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            iter_data_time = time.time()

        if rank == 0:
            if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                model.save_networks('latest')
                model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.

    dist.destroy_process_group()

if __name__ == '__main__':

    opt = TrainOptions().parse()   # get training options

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in opt.gpu_ids)

    world_size = len(opt.gpu_ids)

    mp.spawn(train_main, args=(world_size, opt,), nprocs=world_size, join=True)


