import time

import requests
import sglang as sgl
import torch


def main():

    model_name = 'Qwen/Qwen2.5-0.5B'
    base_url = 'http://localhost:30000'

    generate_url = f"{base_url}/generate"
    release_url = f"{base_url}/release_memory_occupation"
    resume_url = f"{base_url}/resume_memory_occupation"
    update_url = f"{base_url}/update_weights_from_disk"
    session = requests.Session()
    session.headers.update({'Content-Type': 'application/json'})

    def call_generate():
        for _ in range(5):
            payload = {
                'text': [
                    'Hello, my name is',
                    'The president of the United States is',
                    'The capital of France is',
                    'The future of AI is',
                ],
                'sampling_params': {'temperature': 0.8, 'top_p': 0.95},
            }

            response = session.post(generate_url, json=payload)
            response.raise_for_status()

            _ = response.json()

    def call_release():
        response = session.post(release_url, json={})
        response.raise_for_status()

        return response.json()

    def call_resume():
        response = session.post(resume_url, json={})
        response.raise_for_status()

        return response.json()

    def call_update():
        response = session.post(update_url, json={'model_path': model_name})
        response.raise_for_status()

        return response.json()

    for i in range(100):
        print(f"Start step {i}...")

        call_generate()

        call_release()

        time.sleep(30)

        call_resume()

        call_update()


if __name__ == '__main__':
    main()
