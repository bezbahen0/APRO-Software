import argparse
import model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint


class Config:
    batch_size = 32
    num_epochs = 10
    seq_length = 50
    use_words = False
    
    save_path = './data/'

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  mode='auto',
                                  factor=0.8,
                                  patience=2,
                                  epsilon=1e-4,
                                  coldown=5,
                                  min_lr=1e-5)
    
    checkpoint_best = ModelCheckpoint(save_path + 'best_model.h5',
                                      monitor='val_loss',
                                      mode='min',
                                      verbose=1,
                                      save_best_only=True,
                                      save_weights_only=False)
    
    checkpoint_last = ModelCheckpoint(save_path + 'last_model.h5',
                                      monitor='val_loss',
                                      mode='min',
                                      verbose=1,
                                      save_best_only=False,
                                      save_weights_only=False)
    
    
    callbacks = [reduce_lr, checkpoint_best, checkpoint_last]

def train(model):
    pass

if __name__ == "__main__":
    config = Config()
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--text", action="store", required=False, dest="text", help="Input text file")
    parser.add_argument("-e", "--epochs", action="store", required=False, dest="epochs", help="Number of training epochs")
    parser.add_argument("-b", "--batch_size", action="store", required=False, dest="batch_size", help="Size of batch")
    parser.add_argument("-w", "--words_use", action="store", required=False, dest="use_words", help="Words using False/True")
    parser.add_argument("-s", "--save_path", action="store", required=False, dest="save_path", help="Path to save model weigths")
    
    
    args = parser.parse_args()
    
    if args.batch_size is not None:
        config.batch_size = int(args.batch_size)
    if args.epochs is not None:
        config.num_epochs = int(args.epochs)
    if args.phrase_len is not None:
        config.seq_length = int(args.phrase_len)
    if args.use_words is not None:
        config.use_words = bool(args.use_words)
    if args.save_path is not None:
        config.save_path = args.save_path

    model = model.make_text_generator_model()

    train(model)



