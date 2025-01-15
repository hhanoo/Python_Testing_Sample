import ollama
from pathlib import Path

PROMPT = """You are assisting a robot performing an assembly task.

            The first picture is before you perform one assembly.
            The second picture is after one assembly.
            
            Task List:
            pick_and_place()
            bolting()
            combine()
            
            Additional explain:
            Place a new object in the first image, this task is pick_and_place( ).
            If you are using a bolt, this task is to secure the () with a bolt.
            There are two big objects in the first image, one object in the second image, and this is a combined ()
            
            Based on the given image, identify the required action in the list above.  
            The image only performs one operation.
            Please let me know the function name of the most appropriate task without explanation.
            """


def main():
    folder_path = Path("../media/llava")

    file_list = [file for file in folder_path.iterdir() if file.is_file()]

    for i in range(len(file_list) - 1):
        if i == 2 or i == 4:
            continue
        print(file_list[i])
        response = ollama.chat(
            model='llava:34b',
            messages=[{
                'role': 'user',
                'content': PROMPT,
                'images': [file_list[i], file_list[i + 1]],
            }],
            options={
                'seed': 0,
                'temperature': 0
            }
        )

        print(response['message']['content'])
        print()


if __name__ == "__main__":
    main()
