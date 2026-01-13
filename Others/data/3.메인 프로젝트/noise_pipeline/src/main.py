from src.fetcher import fetch_all_noise_data
from src.processor import filter_region, to_dataframe
from src.kafka_producer import get_producer, send_noise_data


def run_pipeline():
    all_data = fetch_all_noise_data()
    filtered_data = filter_region(all_data)

    producer = get_producer()
    send_noise_data(producer, filtered_data)

    df = to_dataframe(filtered_data)
    df.to_csv("data/raw/gangnam_noise_data.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    run_pipeline()
