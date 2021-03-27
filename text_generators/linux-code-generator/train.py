import time
import argparse
from model import model
from data import load_data
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint


class Config:
    batch_size = 32
    num_epochs = 10
    seq_length = 50
    validaion_rate = 0.2
    use_words = False

    
    save_path = './weights/'
    data_path = './data/'

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
                                      mode='min', verbose=1, save_best_only=False,
                                      save_weights_only=False)
    
    
    callbacks = [reduce_lr, checkpoint_best, checkpoint_last]

def train(model, x_train, y_train):
    print('[INFO] Start training model')

    train_start = time.time()
    
    model.fit(x_train,
              y_train,
              validation_rate=config.validation_rate,
              batch_size=config.batch_size,
              epochs=config.num_epochs,
              verbose=1,
              callbaks=config.callbacks)

    train_end = time.time()
    print('[INFO] End training model')
    print('[INFO] Training time: ', train_end - train_start)

if __name__ == "__main__":
    config = Config()
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--text", action="store", required=False, dest="text", help="Input text file")
    parser.add_argument("-e", "--epochs", action="store", required=False, dest="epochs", help="Number of training epochs")
    parser.add_argument("-b", "--batch_size", action="store", required=False, dest="batch_size", help="Size of batch")
    parser.add_argument("-w", "--words_use", action="store", required=False, dest="use_words", help="Words using False/True")
    parser.add_argument("-s", "--save_path", action="store", required=False, dest="save_path", help="Path to save model weigths")
    parser.add_argument("-v", "--validaion_rate", action="store", required=False, dest="validaion_rate", help="Rate validation data")
    

    
    args = parser.parse_args()
    
    if args.batch_size is not None:
        config.batch_size = int(args.batch_size)
    if args.epochs is not None:
        config.num_epochs = int(args.epochs)
    if args.use_words is not None:
        config.use_words = bool(args.use_words)
    if args.save_path is not None:
        config.save_path = args.save_path
    if args.validaion_rate is not None:
        config.validation_rate = args.validation_rate

    x_train, y_train, vectorizer = load_data.load_data(args.text, config.use_words, False, False, config.batch_size, config.seq_length)
    

    model = model.make_text_generator_model()


    train(model, x_train, y_train)



