class Config(object):
    learning_rate = 0.01
    epsilon = 1e-8
    max_grad_norm = 40.0
    evaluation_interval = 10
    batch_size = 32
    hops = 3
    epochs = 100
    embedding_size = 40
    memory_size = 50
    random_state = None
    # set the task ids to train
    task_ids = range(1,21)
    # dataset dir
    data_dir = 'tasks_1-20_v1-2/en/'
    # output file
    output_file = 'results.csv'
    val_size = 0.1
    stddev = 0.1
    learning_rate = 1e-2
    name ='memory'
    # dir for saving and restoring
    save_file = './weights/mem00'
    load_file = './weights/mem00'
