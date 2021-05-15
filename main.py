import pandas as pd
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--consumption", default="./sample_data/consumption.csv", help="input the consumption data path")
    parser.add_argument("--generation", default="./sample_data/generation.csv", help="input the generation data path")
    parser.add_argument("--bidresult", default="./sample_data/bidresult.csv", help="input the bids result path")
    parser.add_argument("--output", default="output.csv", help="output the bids path")

    return parser.parse_args()


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))

        return predictions[-1]


def normalize(data):
    '''
    x_norm = (x-xmin)/(xmax-xmin)
    '''
    data_norm = data.copy()
    data_norm = (data_norm - data.min())/(data.max()-data.min())
    data_norm = 2*data_norm - 1

    return data_norm, data.max(), data.min()


def inverse_normalize(data, data_max, data_min):
    data = (data+1)/2*(data_max-data_min)+ data_min   

    return data


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq


def preprocessing(training_data, train_window = 24):
    train_norm, _, _ = normalize(training_data)
    train_norm = torch.FloatTensor(train_norm).view(-1)
    train_inout_seq = create_inout_sequences(train_norm, train_window)

    return train_inout_seq


def load_predict_data(con_file, gen_file):
    df_c = pd.read_csv(con_file)
    df_g = pd.read_csv(gen_file)

    # consumption - generation
    df_retrain = df_c
    df_retrain['c_g'] = df_c['consumption'] - df_g['generation']

    df_retrain = df_retrain['c_g']


    df_retrain = np.array(df_retrain)

    return df_retrain 


def predict(seq):
    # load model
    model = LSTM()
    model.load_state_dict(torch.load('./model'))
    model.eval()
    
    # normalize
    seq, seqmax, seqmin = normalize(seq)

    # predict
    row_norm = torch.FloatTensor(seq)
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size)) 
        actual_predictions = inverse_normalize(np.array(model(row_norm).item()), seqmax, seqmin) 

    seq_pre = []
    for i in range(24):
        row = seq[i:i+24]
        row_norm = torch.FloatTensor(row)
        with torch.no_grad():
            model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size)) 
            actual_predictions = inverse_normalize(np.array(model(row_norm).item()), seqmax, seqmin)
            seq_pre.append(actual_predictions) 
    
    return np.squeeze(seq_pre)



def algo_trade(seq_predict, con_file):
    # initialize dataframe
    df_trade = pd.DataFrame(columns=['time','action','target_price','target_volume'])

    # get timestamp
    df_c = pd.read_csv(con_file)
    last_t = df_c['time'][-1:]
    last_t = np.array(last_t)[0]
    time=pd.date_range(last_t, periods=25, freq='H', name='time')[1:]

    # save trade in df_trade
    df_trade = pd.DataFrame(columns=['time','action','target_price','target_volume'])

    i = 0
    for num, val in enumerate(seq_predict):
        if val >0:
            df_trade.loc[i] = [time[num]] + ['buy'] + [str(2.52)] +  [str(round(val, 2))]
            i += 1
        elif val <0:
            df_trade.loc[i] = [time[num]] + ['sell'] + [str(2.54)] +  [str(round(abs(val), 2))]
            i += 1

    return df_trade


def output(path, df):
    df.to_csv(path, index=False)

    return 


if __name__ == '__main__':
    args = config()

    # load testing data
    testing_data = load_predict_data(args.consumption, args.generation)

    # predict
    seq_predict = predict(testing_data)
    
    # decide trade actions
    df_trade = algo_trade(seq_predict, args.consumption)

    # save output file  
    output(args.output, df_trade)