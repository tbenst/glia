import optuna, os

pw = os.environ["POSTGRES_OPTUNA_PASSWORD"]
server = os.environ["POSTGRES_SERVER"]
port = os.environ["POSTGRES_PORT"]
user = os.environ["POSTGRES_OPTUNA_USER"]

study_name = "2020-09-14_FEI_VAE"
storage = f'postgresql://{user}:{pw}@{server}:{port}/optuna'

optuna.delete_study(study_name, storage=storage)