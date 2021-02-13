import argparse
import logging
import os

import torch
import torch.distributed as dist

from ssd.engine.inference import do_evaluation
from ssd.config import cfg
from ssd.data.build import make_data_loader
from ssd.engine.trainer import do_train
from ssd.modeling.detector import build_detection_model
from ssd.solver.build import make_optimizer, make_lr_scheduler
from ssd.utils import dist_util, mkdir
from ssd.utils.checkpoint import CheckPointer
from ssd.utils.dist_util import synchronize
from ssd.utils.logger import setup_logger
from ssd.utils.misc import str2bool


def train(cfg, args):
    logger = logging.getLogger('SSD.trainer')
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)


    lr = cfg.SOLVER.LR * args.num_gpus  # scale by num gpus
    logger.info('Variable lr: {}'.format(lr))
    optimizer = make_optimizer(cfg, model, lr)

    milestones = [step // args.num_gpus for step in cfg.SOLVER.LR_STEPS]
    scheduler = make_lr_scheduler(cfg, optimizer, milestones)
    logger.info('Learning Rate {}'.format(scheduler.get_lr()))

    arguments = {"iteration": 0}
    save_to_disk = dist_util.get_rank() == 0

    checkpointer = CheckPointer(model, optimizer, scheduler, cfg.OUTPUT_DIR, save_to_disk, logger)
    extra_checkpoint_data = checkpointer.load(args.ckpt, use_latest=args.ckpt is None)
    arguments.update(extra_checkpoint_data)
    weight_file = args.ckpt if args.ckpt else checkpointer.get_checkpoint_file()
    logger.info('Loaded weights from {}'.format(weight_file))

    '''
    IF A PRETRAINED MODEL IS USED THEN THE ORIGINAL CODE OVERRIDES LEARNING RATE AND SCHEDULER
    INFORMATION. THIS CODE SNIPPET ALLOWS US TO RELOAD OUR OWN CONFIGURATION
    '''
    if args.ckpt is not None:
      optimizer = make_optimizer(cfg, model, lr)
      logger.info("Loading optimizer from {}".format(args.ckpt))
      scheduler = make_lr_scheduler(cfg, optimizer, milestones)
      logger.info("Loading scheduler from {}".format(args.ckpt))
      checkpointer.scheduler_opt(scheduler, optimizer)
      
    logger.info('After Checkpoint Learning Rate {}'.format(scheduler.get_lr()))

    max_iter = cfg.SOLVER.MAX_ITER // args.num_gpus
    train_loader = make_data_loader(cfg, is_train=True, distributed=args.distributed, max_iter=max_iter, start_iter=arguments['iteration'], isStyle = False)
    
    '''
    IF AN ADAIN MODEL IS SPECIFIED IT IS NECESSARY TO LOAD A STYLE_LOADER WHICH IS COMPRISED OF STYLE IMAGES TO BE USED DURING TRAINING
    '''
    if args.AdaIN_model != "None":
      style_loader = make_data_loader(cfg, is_train=True, distributed=args.distributed, max_iter=max_iter, start_iter=arguments['iteration'], isStyle = True)
    else:
      style_loader = None
    model = do_train(cfg, model, train_loader, style_loader, optimizer, scheduler, checkpointer, device, arguments, args)
    return model


def main():
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With PyTorch')
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--log_step', default=10, type=int, help='Print logs every log_step')
    parser.add_argument('--save_step', default=2500, type=int, help='Save checkpoint every save_step')
    parser.add_argument('--eval_step', default=2500, type=int, help='Evaluate dataset every eval_step, disabled when eval_step < 0')
    parser.add_argument('--use_tensorboard', default=True, type=str2bool)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    '''
    ADDED COMMAND LINE ARGUMENTS
    ADAIN_MODEL TAKES PATH TO AN ADAIN MODEL
    CKPT LOADS A PRETRAINED SSD MODEL
    '''
    parser.add_argument(
        "--AdaIN_model",
        default="None",
        metavar="FILE",
        help="path to AdaIN model file",
        type=str,
    )
    parser.add_argument("--ckpt", type=str, default=None, help="Trained weights.")
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    args = parser.parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    args.num_gpus = num_gpus

    if torch.cuda.is_available():
        # This flag allows you to enable the inbuilt cudnn auto-tuner to
        # find the best algorithm to use for your hardware.
        torch.backends.cudnn.benchmark = True
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if cfg.OUTPUT_DIR:
        mkdir(cfg.OUTPUT_DIR)

    logger = setup_logger("SSD", dist_util.get_rank(), cfg.OUTPUT_DIR)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
  
    model = train(cfg, args)

    if not args.skip_test:
        logger.info('Start evaluating...')
        torch.cuda.empty_cache()  # speed up evaluating after training finished
        do_evaluation(cfg, model, distributed=args.distributed)


if __name__ == '__main__':
    main()
