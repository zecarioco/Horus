from backend.data_manager import DataManager
from backend.trainer_worker import TrainerWorker

def run_training(
    max_length=128,
    epochs=3,
    lr=2e-6, 
    batch_size=8,
    display_name=None,
    model_name="neuralmind/bert-base-portuguese-cased",
    fp16=False, 
):
    dm = DataManager(model_name=model_name)

    data = dm.run(max_length=max_length)

    trainer = TrainerWorker(
        display_name=display_name,
        learning_rate=lr,
        epochs=epochs,
        batch_size=batch_size,
        model_name=model_name,
        fp16=fp16 
    )

    metrics = trainer.train(
        train_dataset=data["train_dataset"],
        test_dataset=data["test_dataset"],
        tokenizer=data["tokenizer"],
        num_labels=len(data["label_cols"]),
        label_cols=data["label_cols"]
    )

    return metrics