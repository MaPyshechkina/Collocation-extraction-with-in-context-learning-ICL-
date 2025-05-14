import pandas as pd
import yaml
from yandex_cloud_ml_sdk import YCloudML
from response_cleaner import ResponseCleaner  # Импортируем процессор

def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def load_prompt(prompt_path):
    with open(prompt_path, "r", encoding="utf-8") as file:
        return file.read().strip()

def process_text_file(config):
    input_file = config["input_file"]
    output_file = config["output_file"]
    folder_id = config["folder_id"]
    auth = config["auth"]
    model = config["model"]
    temperature = config["temperature"]
    target_word = config["target_word"]
    prompt_text = load_prompt(config["prompt_file"])

    with open(input_file, "r", encoding="utf-8") as file:
        sentences = [line.strip() for line in file if line.strip()]

    results_df = pd.DataFrame(columns=["Sentence", "Collocation", "Cleaned"])

    for sentence in sentences:
        messages = [
            {"role": "system", "text": prompt_text},
            {
                "role": "user",
                "text": f"Найди коллокацию со словом '{target_word}' в предложении: {sentence}"
            },
        ]

        try:
            sdk = YCloudML(folder_id=folder_id, auth=auth)

            result = (
                sdk.models.completions(model)
                .configure(temperature=temperature)
                .run(messages)
            )

            collocation = result[0].text.strip() if result else "Не удалось определить"
            cleaned = ResponseCleaner.clean(collocation)

        except Exception as e:
            print(f"Ошибка обработки предложения: {sentence}\n{e}")
            collocation = "Ошибка"
            cleaned = "Ошибка"

        results_df = pd.concat(
            [results_df, pd.DataFrame([{
                "Sentence": sentence,
                "Collocation": collocation,
                "Cleaned": cleaned
            }])],
            ignore_index=True,
        )

    try:
        old_results = pd.read_csv(output_file)
        results_df = pd.concat([old_results, results_df], ignore_index=True)
    except FileNotFoundError:
        pass

    results_df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Результаты сохранены в файл: {output_file}")

if __name__ == "__main__":
    config = load_config()
    process_text_file(config)
