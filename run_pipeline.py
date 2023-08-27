from pipelines.training_pipeline import training_model
from zenml.client import Client


if __name__ == "__main__":
    #run the pipeline
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    training_model(data_path="C:/Users/pc/Documents/Mlops project/data1/olist_customers_dataset.csv")

   