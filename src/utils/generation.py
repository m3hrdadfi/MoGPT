import json
import os
import sys


def generation(data_loader, model, tokenizer, generation_kwargs, args, wandb_logger=None, bar_length=30):
    data_table = []
    counter = 1
    total = len(data_loader)

    for idx, batch in enumerate(data_loader):
        idx = idx + 1
        percent = 100.0 * idx / total
        sys.stdout.write("\r")
        sys.stdout.write("Completed: [{:{}}] {:>3}%".format('=' * int(percent / (100.0 / bar_length)), bar_length, int(percent)))

        for i in range(batch["input_ids"].shape[0]):
            batch_input = {
                k: v[i][v[i] != tokenizer.pad_token_id].unsqueeze(dim=0).to(model.device)
                if k == "input_ids" else
                v[i].unsqueeze(dim=0).to(model.device)
                for k, v in batch.items()
            }
            response = batch_input["labels"].squeeze(dim=0).to(model.device)
            del batch_input["labels"]

            g_responses = model.model.generate(**batch_input, **generation_kwargs)

            for j, g_response in enumerate(g_responses):
                data_table.append([
                    f"epoch_end-item_{counter}-gen_{j + 1}",
                    tokenizer.decode(batch_input["input_ids"][0]).replace(tokenizer.pad_token, "").strip(),
                    tokenizer.decode(response[response > -1]),
                    tokenizer.decode(g_response[len(batch_input["input_ids"][0]):]).replace(tokenizer.pad_token, "").strip()
                ])

            counter += 1

        sys.stdout.flush()

    output_json = os.path.join(args.output_dir, "outputs")
    os.makedirs(output_json, exist_ok=True)
    with open(os.path.join(output_json, f"epoch_end.json"), "w") as fj:
        json.dump({"output": data_table}, fj)

    if wandb_logger:
        wandb_logger.log_table(
            key=f"epoch_end",
            columns=["state", "dialog", "r_response", "g_response"],
            data=data_table
        )
