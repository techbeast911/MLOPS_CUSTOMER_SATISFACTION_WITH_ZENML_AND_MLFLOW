from zenml import pipeline
from steps.clean_data import clean_df
from steps.evaluation import evaluation
from steps.ingest_data import ingest_df
from steps.model_train import train_model


@pipeline(enable_cache=True)
def training_model(data_path: str):
    df= ingest_df(data_path)
    X_train,X_test,y_train,y_test= clean_df(df)
    model=train_model(X_train,X_test,y_train,y_test)
    r2_score,rmse=evaluation(model,X_test,y_test)
    