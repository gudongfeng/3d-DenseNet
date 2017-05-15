import sys
import argparse
import numpy as np

from models.dense_net_3d import DenseNet3D
from data_providers.utils import get_data_provider_by_name

train_params_merl = {
  'num_classes': 5,
  'batch_size': 10,
  'n_epochs': 100,
  'crop_size': 100,
  'sequence_length': 16,
  'initial_learning_rate': 0.1,
  'reduce_lr_epoch_1': 50,  # epochs * 0.5
  'reduce_lr_epoch_2': 75,  # epochs * 0.75
  'validation_set': True,
  'validation_split': None,  # None or float
  'normalization': 'std',  # None, divide_256, divide_255, std
}

train_params_ucf101 = {
  'num_classes': 101,
  'batch_size': 64,
  'n_epochs': 40,
  'crop_size': 64,
  'sequence_length': 16,
  'initial_learning_rate': 0.1,
  'reduce_lr_epoch_1': 20,
  'reduce_lr_epoch_2': 30,
  'validation_set': True,
  'validation_split': None,  # you may set it 6000 as in the paper
  'normalization': 'divide_255',
}


def get_train_params_by_name(name):
  if name == 'UCF101':
    return train_params_ucf101
  if name == 'MERL':
    return train_params_merl


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--train', action='store_true',
    help='Train the model')
  parser.add_argument(
    '--test', action='store_true',
    help='Test model for required dataset if pretrained model exists.'
       'If provided together with `--train` flag testing will be'
       'performed right after training.')
  parser.add_argument(
    '--model_type', '-m', type=str, choices=['DenseNet', 'DenseNet-BC'],
    default='DenseNet',
    help='What type of model to use (default: %(default)s)')
  parser.add_argument(
    '--growth_rate', '-k', type=int, choices=[12, 24, 40],
    default=12,
    help='Grows rate for every layer, '
       'choices were restricted to used in paper (default: %(default)s)')
  parser.add_argument(
    '--depth', '-d', type=int, choices=[20, 30, 40, 100, 190, 250],
    default=20,
    help='Depth of whole network, restricted to paper choices (default: %(default)s)')
  parser.add_argument(
    '--dataset', '-ds', type=str,
    choices=['MERL', 'UCF101'],
    default='MERL',
    help='What dataset should be used (default: %(default)s)')
  parser.add_argument(
    '--total_blocks', '-tb', type=int, default=3, metavar='',
    help='Total blocks of layers stack (default: %(default)s)')
  parser.add_argument(
    '--keep_prob', '-kp', type=float, metavar='',
    help="Keep probability for dropout.")
  parser.add_argument(
    '--gpu_id', '-gid', type=int, default=0,
    help='Specify the gpu ID to run the program')
  parser.add_argument(
    '--weight_decay', '-wd', type=float, default=1e-4, metavar='',
    help='Weight decay for optimizer (default: %(default)s)')
  parser.add_argument(
    '--nesterov_momentum', '-nm', type=float, default=0.9, metavar='',
    help='Nesterov momentum (default: %(default)s)')
  parser.add_argument(
    '--reduction', '-red', type=float, default=0.5, metavar='',
    help='reduction Theta at transition layer for DenseNets-BC models (default: %(default)s)')

  parser.add_argument(
    '--logs', dest='should_save_logs', action='store_true',
    help='Write tensorflow logs')
  parser.add_argument(
    '--no-logs', dest='should_save_logs', action='store_false',
    help='Do not write tensorflow logs')
  parser.set_defaults(should_save_logs=True)

  parser.add_argument(
    '--saves', dest='should_save_model', action='store_true',
    help='Save model during training')
  parser.add_argument(
    '--no-saves', dest='should_save_model', action='store_false',
    help='Do not save model during training')
  parser.set_defaults(should_save_model=True)

  parser.add_argument(
    '--renew-logs', dest='renew_logs', action='store_true',
    help='Erase previous logs for model if exists.')
  parser.add_argument(
    '--not-renew-logs', dest='renew_logs', action='store_false',
    help='Do not erase previous logs for model if exists.')
  parser.set_defaults(renew_logs=False)

  args = parser.parse_args()

  if not args.keep_prob:
    if args.dataset in ['UCF101']:
      args.keep_prob = 0.8
    else:
      args.keep_prob = 1.0
  if args.model_type == 'DenseNet':
    args.bc_mode = False
    args.reduction = 1.0
  elif args.model_type == 'DenseNet-BC':
    args.bc_mode = True

  model_params = vars(args)

  if not args.train and not args.test:
    print("You should train or test your network. Please check params.")
    exit()

  # ==========================================================================
  # LOG FILE SETTING
  # ==========================================================================
  # write all the log to the file without buffer
  f = open('log.txt', 'w', 0)
  sys.stdout = f
  sys.stderr = f

  # ==========================================================================
  # PARAMETERS PRINTING
  # ==========================================================================
  # some default params dataset/architecture related
  train_params = get_train_params_by_name(args.dataset)
  print("Params:")
  for k, v in model_params.items():
    print("\t%s: %s" % (k, v))
  print("Train params:")
  for k, v in train_params.items():
    print("\t%s: %s" % (k, v))
  
  # ==========================================================================
  # DATA PREPARATION
  # ==========================================================================
  train_params['test'] = args.test
  data_provider = get_data_provider_by_name(args.dataset, train_params)

  # ==========================================================================
  # TRAINING & TESTING
  # ==========================================================================
  print("Initialize the model..")
  model = DenseNet3D(data_provider=data_provider, **model_params)
  if args.train:
    print("Data provider train videos: ", data_provider.train.num_examples)
    model.train_all_epochs(train_params)
  if args.test:
    if not args.train:
      model.load_model()
    print("Data provider test videos: ", data_provider.test.num_examples)
    print("Testing...")
    losses = []
    accuracies = []
    for i in range(10):
      loss, accuracy = model.test(data_provider.test, batch_size=10)
      losses.append(loss)
      accuracies.append(accuracy)
    loss     = np.mean(losses)
    accuracy = np.mean(accuracies)
    print("mean cross_entropy: %f, mean accuracy: %f" % (loss, accuracy))

