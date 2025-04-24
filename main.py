import os 
import glob
import pickle
import torch
import numpy as np 
from random import shuffle
from torch.optim import Adam
import torch.nn.functional as F
from pathlib import Path
from utils.scalers import FeatureScaler
from graph_build.gb import Graph_build
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
# from charge_prediction_system import *
from model_rh import *
from matplotlib import pyplot as plt
import math
import copy
import torch
# from src.models.gnn import GNNModel
# from src.utils.scalers import StandardScaler

def train_test():
    pass
def iterate_cif_files(directory):
 
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"The directory {directory} does not exist.")

    # Construct the glob pattern to match all .cif files
    pattern = os.path.join(directory, '*.cif')

    # Use glob to find all matching CIF files
    cif_files = glob.glob(pattern)

    if not cif_files:
        print(f"No CIF files found in directory: {directory}")
        return

    print(f"Found {len(cif_files)} CIF files in directory: {directory}")
    return cif_files
def graph_ftrs_write(root_path):
    # will read all the CIF call the graph_build.gb and return Graph data ... we pickle them in data/graphs
    cif_files=iterate_cif_files(root_path)
    output_dir = 'data/graphs'
    #parallelize this part 
    f = open("data/graphs/crystal_names.txt", "w")
    for k in cif_files:   
        name=k.split('/')[-1]
        graph_dataset = Graph_build(k)
        node_features,charges,edges= graph_dataset.get_data()
        f.write(f"{name}\n")
        # Save all outputs to a subdirectory
        node_features_file = os.path.join(output_dir, f"{name}_node_ftrs.pkl")
        charges_file = os.path.join(output_dir, f"{name}_charges.pkl")
        edges_file = os.path.join(output_dir, f"{name}_edges.pkl")

        # Save each numpy array to its respective pickle file
        with open(node_features_file, 'wb') as nf:
            pickle.dump(node_features, nf)
        with open(charges_file, 'wb') as ch:
            pickle.dump(charges, ch)
        with open(edges_file, 'wb') as ed:
            pickle.dump(edges, ed)
    f.close()
# loading the written files            
def load_graph_data():
    #loads the pickled graph data in data/graphs
    names=np.loadtxt('data/graphs/crystal_names.txt',dtype=str)
    print(names)
    all_nodes=[]
    all_charges=[]
    all_edges=[]
    for name in names:
        directory='data/graphs/'
        node_features_file = os.path.join(directory, f"{name}_node_ftrs.pkl")
        charges_file = os.path.join(directory, f"{name}_charges.pkl")
        edges_file = os.path.join(directory, f"{name}_edges.pkl")

        # Check if the files exist
        if not all(os.path.isfile(file) for file in [node_features_file, charges_file, edges_file]):
            raise FileNotFoundError(f"Pickle files for {name} not found in {directory}.")

        # Load data from pickle files
        with open(node_features_file, 'rb') as nf:
            node_features = pickle.load(nf)
            node_features=torch.tensor(node_features, dtype=torch.double)
        all_nodes.append(node_features)

        with open(charges_file, 'rb') as ch:
            charges = pickle.load(ch)
            charges=torch.tensor(charges, dtype=torch.double)
        all_charges.append(charges)

        with open(edges_file, 'rb') as ed:
            edges = pickle.load(ed)
            edges=torch.tensor(edges, dtype=torch.long)
        all_edges.append(edges)

    return all_nodes,all_charges,all_edges


def train(train_data,val_data,all_nodes):
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    print("print train loader")
    #print(train_data)
    valid_loader = DataLoader(val_data,batch_size=1,shuffle=False)
    MAX_EPOCHS = 30
    BATCH_SIZE = 1
    MAX_ITERATIONS = 1

    GNN_LAYERS = 4
    EMBEDDING_SIZE = 20
    HIDDEN_FEATURES_SIZE = 30
    iteration = 0
    # GRAPHS_LOCATION = "input"
    # data_list = data_handling(GRAPHS_LOCATION, READ_LABELS = True)
    # NUM_NODE_FEATURES = data_list[0]['x'].shape[1]
    NUM_NODE_FEATURES = len(all_nodes[1][0])
    print(NUM_NODE_FEATURES)
    # crit =
    train_data_size = len(train_data)
    valid_data_size = len(val_data)
    crit = torch.nn.L1Loss()
    
    # systems = ['vanilla', 'soft_con', 'mean_cor', 'gaussian_cor']
    #systems = [ 'mean_cor', 'gaussian_cor']
    systems = [ 'mean_cor']
    models = []
    iteration=0

    for system in systems:
        model = charge_prediction_system(train_loader, valid_loader,NUM_NODE_FEATURES,EMBEDDING_SIZE,GNN_LAYERS,HIDDEN_FEATURES_SIZE,train_data_size, valid_data_size, MAX_EPOCHS, iteration, system, crit)
        models.append(model)

    print("training_working")
    return models


def charge_prediction_system(train_loader, valid_loader,NUM_NODE_FEATURES,EMBEDDING_SIZE,GNN_LAYERS,HIDDEN_FEATURES_SIZE,train_data_size, valid_data_size, MAX_EPOCHS, iteration, system, crit = torch.nn.L1Loss()):
    # initializing the model

    device = torch.device('cpu')
    if (system == 'vanilla' or system == 'soft_con'):
        print(">>> vanilla model")
        model = Net_vanilla(NUM_NODE_FEATURES,EMBEDDING_SIZE,GNN_LAYERS,HIDDEN_FEATURES_SIZE).to(device)
    elif(system == 'mean_cor'):
        print(">>> mean_correction_model")
        model = Net_mean_correction(NUM_NODE_FEATURES,EMBEDDING_SIZE,GNN_LAYERS,HIDDEN_FEATURES_SIZE).to(device)
    # else:
    #     print(">>> gaussian_correction_model")
    #     model = Net_gaussian_correction(NUM_NODE_FEATURES,EMBEDDING_SIZE,GNN_LAYERS,HIDDEN_FEATURES_SIZE).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    model = model.double()

    train_total_loss = []
    valid_total_loss = []
    min_valid_loss = float("inf")
    rebound = 0 # number of epochs that validation loss is increasing
    for epoch in range(MAX_EPOCHS):
        model.train()
        loss_all = 0

#######################################################################     
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            print(data)
            # print(data)
        #   data = torch.tensor(data, dtype=torch.float32)
            pred, _ , _, _= model(data)

            label = data.y.to(device)
            if (system == 'soft_con'):
                # MAE loss plus total sum of predicted values --since it should be zero 
                loss = 100*crit(pred, label) + abs(sum(pred))
            else:
                loss = crit(pred, label)
            loss.backward()
            loss_all += data.num_graphs * loss.item()
            optimizer.step()
#           print("||| PREDICTION SUM FOR ONE MOF: ", torch.sum(pred ))
        loss_epoch = loss_all / train_data_size

        # evaluating model
        model.eval()
        loss_all = 0
        with torch.no_grad():
            for data in train_loader:
                data = data.to(device)
                pred, _, _,_  = model(data)
                label = data.y.to(device)
                if (system == 'soft_con'):
                    loss = 100 * crit(pred, label) + abs(sum(pred))
                else:
                    loss = crit(pred, label)
                loss_all += data.num_graphs * loss.item()
        train_acc = loss_all / train_data_size
        train_total_loss.append(train_acc)
        # evaluating valid dataset
        model.eval()
        loss_all = 0
        with torch.no_grad():
            for data in valid_loader:
                data = data.to(device)
                pred, _, _, _ = model(data)
                label = data.y.to(device)
                if (system == 'soft_con'):
                    loss = 100 * crit(pred, label) + abs(sum(pred))
                else:
                    loss = crit(pred, label)
                loss_all += data.num_graphs * loss.item()
        valid_acc = loss_all / valid_data_size

        valid_total_loss.append(valid_acc)
        if valid_acc <= min_valid_loss: # keep tracking of model with lowest validation loss
            torch.save(model.state_dict(), './results/loss_iteration_' + str(iteration)+'_system_' + system + '.pth')
            min_valid_loss = valid_acc
            model_min = copy.deepcopy(model)
            rebound = 0
        else:
            rebound += 1
        if(epoch%10==0):
            print('Epoch: {:03d}, Loss: {:.5f}, train_loss: {:.5f}, valid_loss: {:.5f}'.
              format(epoch+1, loss_epoch, train_acc, valid_acc))

        if rebound > 100: # early stopping criterion
            break
#           pass

    hfont = {'fontname':'DejaVu Sans'}
    fontsize_label_legend = 24
    plt.figure(figsize=(8,8), dpi= 80)
    plt.plot(train_total_loss, label="train loss", color='dodgerblue', linewidth = 1.5)
    plt.plot(valid_total_loss, label="valid loss", color='red', linewidth = 1.5)
    plt.legend(frameon=False, prop={'size': 22})
    plt.xlabel('Epochs', fontsize=fontsize_label_legend, **hfont)
    plt.ylabel('Loss', fontsize=fontsize_label_legend, **hfont)
    plt.legend(frameon=False, prop={"family":"DejaVu Sans", 'size': fontsize_label_legend})
    plt.tick_params(axis='both', which='major', labelsize=17)
    plt.savefig('./results/loss_iteration_' + str(iteration)+'_system_' + system+'.png')
    plt.show()

    return model_min


def main():
    # get the data directory /data/ddec_xtals  supply this path to graph_build.py 
    #root_path = Path('data/ddec_xtal_1/ABAVIJ_clean.cif')
    #root_path = Path('data/ddec_xtal_1/LiTiS2.cif')

    #preprocessing
    root_path = Path('data/ddec_xtal_1')
    # writing the graph data 
    graph_ftrs_write(root_path)

    # load the graph data 
    all_nodes,all_charges,all_edges=load_graph_data()
    print(np.shape(all_nodes[1]))
    #scaling 
    scaler = FeatureScaler(method='standard')
    scaler.fit(all_nodes)
    #preaparing the training data  
    data_list=[]
    for k in zip(all_nodes,all_charges,all_edges):
        x = scaler.transform(k[0])
        x=torch.tensor(x, dtype=torch.double)
        data_list.append(Data(x=x, y=k[1], edge_index=k[2]))

    # Split the data into training and testing datasets
    train_data, test1_data = train_test_split(data_list, test_size=0.2, random_state=42)
    test_data,val_data = train_test_split(test1_data,test_size=0.4,random_state=42)
    models=train(train_data,val_data,all_nodes)
    
    # Create DataLoaders for  testing sets
    
    # test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    # test_data_size = len(test_dataset)
    # NUM_NODE_FEATURES =
if __name__ == '__main__':
    main()
