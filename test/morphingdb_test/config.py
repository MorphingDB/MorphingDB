db_config = {
    "dbname": "postgres",
    "host": "localhost",
    "port": "5488",
    "user": "postgres",
    "password": "123456"
}

evadb_db_config = """{
    "database": "postgres",
    "host": "localhost",
    "port": "5488",
    "user": "postgres",
    "password": "123456"
}"""


model_prefix = "/home/pgdl/model/"
select_model_prefix = "/home/pgdl/model/select_model/"

# series
slice_model_path = model_prefix + "slice.pt"
swarm_model_path = model_prefix + "swarm.pt"
year_predict_model_path = model_prefix + "year_predict.pt"

# image
cifar10_model_path = model_prefix + "googlenet_cifar10.pt"
imagenet_model_path = model_prefix + "resnet18_imagenet.pt"
stanford_dogs_model_path = model_prefix + "alexnet_stanford_dogs.pt"

# text
financial_phrasebank_model_path = model_prefix + "sentiment_analysis_model.pt"
spiece_model_path = model_prefix + "spiece.model.old"
imdb_model_path = model_prefix + "traced_albert_vec.pt"
sst2_model_path = model_prefix + "traced_albert_vec.pt"


dataset_prefix = "/home/lhh/morphingdb/MorphingDB/test/morphingdb_test/data/"

# series
slice_dataset_path = dataset_prefix + "series/slice/slice_localization_data.csv"
swarm_dataset_path = dataset_prefix + "series/swarm/Swarm_Behaviour.csv"
year_predict_dataset_path = dataset_prefix + "series/yead_predict/YearPredictionMSD.csv"

# image
cifar10_dataset_path = dataset_prefix + "image/cifar10/test/"
imagenet_dataset_path = dataset_prefix + "image/image-net/data/"
stanford_dogs_dataset_path = dataset_prefix + "image/Stanford_Dogs/images/Images/"

# text
financial_phrasebank_dataset_path  = dataset_prefix + "text/financial_phrasebank"
imdb_dataset_path = dataset_prefix + "text/imdb/data/test-00000-of-00001.parquet"
sst2_dataset_path = dataset_prefix + "text/sst2/data/train.tsv"





# evadb model path
evadb_model_prefix = "/home/lhh/morphingdb/MorphingDB/test/morphingdb_test/models/"
evadb_slice_model_path = evadb_model_prefix + "slice.pt"
evadb_swarm_model_path = evadb_model_prefix + "swarm.pt"
evadb_year_predict_model_path = evadb_model_prefix + "year_predict.pt"

evadb_cifar10_model_path = evadb_model_prefix + "googlenet_cifar10.pt"
evadb_imagenet_model_path = evadb_model_prefix + "resnet18_imagenet.pt"
evadb_stanford_dogs_model_path = evadb_model_prefix + "alexnet_stanford_dogs.pt"

evadb_financial_phrasebank_model_path = evadb_model_prefix + "sentiment_analysis_model.pt"
evadb_spiece_model_path = evadb_model_prefix + "spiece.model.old"
evadb_imdb_model_path = evadb_model_prefix + "traced_albert_vec.pt"
evadb_sst2_model_path = evadb_model_prefix + "traced_albert_vec.pt"