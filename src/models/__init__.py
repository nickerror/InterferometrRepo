class Config:
    def __init__(self):
        self.epochs = 5
        self.num_classes = 1
        self.batch_size = 32
        self.learning_rate = 0.01
        self.dataset = "InterferometerPhoto"
        self.architecture = "CNN"
        self.pin_memory = True
        self.momentum = 0.9
        self.step_size = 3
        self.gamma = 0.1
        self.dataset_metadata = "data/raw/1channel/reference/epsilon.csv" # will change for processed
        self.num_workers = 0
        self.data_root_dir = "data/raw/1channel/photo" # will change for processed