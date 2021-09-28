import gtts

import os

script_file = "/home/jose/Documents/video_projects/transform_vectors/script/script.txt"

save_dir = (
    "/home/jose/Documents/video_projects/transform_vectors/media/script_sound_files"
)


def read_script(lines_to_read=None):
    with open(script_file, "r") as script:

        for i_line, line in enumerate(script):

            if line == "" or line == "\n":
                continue

            if lines_to_read:
                if i_line + 1 not in lines_to_read:
                    continue

            print(line)
            current_tts = gtts.gTTS(line)

            current_tts.save(os.path.join(save_dir, line[0:10] + ".mp3"))


if __name__ == "__main__":
    read_script([42, 43])
