import json
import openai
openai.api_base = ""
openai.api_key = ""
import argparse

def just_eval_final_result(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        data_list = json.load(file)

    total_scores = {
        'helpfulness': 0,
        'clarity': 0,
        'factuality': 0,
        'depth': 0,
        'engagement': 0,
    }
    count = 0

    null_judge = False


    for item in data_list[0:800]:
        # test whether parsed_result is null
        if item.get('parsed_result') is None:
            null_judge = True
            prompt = item["prompt"]
            messages = [{"role": "user", "content": prompt}]
            completion = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages
            )
            chat_response = completion
            answer = chat_response.choices[0].message.content

            if answer.startswith("```json"):
                answer = answer[len("```json"):-len("```")]
            
            if answer.startswith("```"):
                answer = answer[len("```"):-len("```")]
                
            print("fulfill id {}".format(item["id"]))
            print(answer)
            item['parsed_result'] = json.loads(answer)

        for aspect in total_scores.keys():
            score = item['parsed_result'].get(aspect, {}).get('score')
            if isinstance(score, str):
                score = int(score)
            total_scores[aspect] += score
        count += 1
    print(count)

    average_scores = {aspect: total / count for aspect, total in total_scores.items()}
    values = list(average_scores.values())
    average = sum(values) / len(values)
    average_scores['Average'] = average
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(average_scores, file, ensure_ascii=False, indent=4)

    if null_judge:
        with open(input_file_path, 'w', encoding='utf-8') as file:
            json.dump(data_list, file, ensure_ascii=False, indent=4)

    print(f"Overall average scores calculated and saved to {output_file_path}")

input_file_path = ""
output_file_path = "average_scores.json"
just_eval_final_result(input_file_path, output_file_path)