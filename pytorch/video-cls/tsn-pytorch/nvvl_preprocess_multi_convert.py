from collections import defaultdict
import os
import time
import multiprocessing
import json

resolution_stat1 = defaultdict(int)
resolution_stat2 = defaultdict(int)


def decode(q):
    while True:
        cmd = q.get()
        os.system(cmd)
        q.task_done()


def get_scale(filename):
    stream_info = os.popen('ffprobe -v quiet -print_format json -show_streams -i "' + filename + '"').read()

    js = json.loads(stream_info)
    scale = '480x360'
    for stream in js['streams']:
        if stream['codec_type'] == 'video':
            width = stream['width']
            height = stream['height']

            if width < height:
                width, height = height, width
                resolution_ratio = round(10.0 * width / height)
                resolution_stat1[resolution_ratio] += 1
                scale = '360x480' if resolution_ratio < 16 else '360x640'
            else:
                resolution_ratio = round(10.0 * width / height)
                resolution_stat2[resolution_ratio] += 1
                scale = '480x360' if resolution_ratio < 16 else '640x360'

            break

    return scale

if __name__ == '__main__':
    cmd_queue = multiprocessing.JoinableQueue(6)

    processes = []
    for i in range(4):
        p = multiprocessing.Process(target=decode, args=(cmd_queue,))
        p.daemon = True
        processes.append(p)

    for p in processes:
        p.start()

    invalid_name = []
    input_file = open("list.txt")
    lines = input_file.readlines()
    start_pos = 0
    for count, line in enumerate(lines[start_pos:]):
        print("#######################Processing {}########################".format(start_pos + count))
        filename = line.strip()
        try:
            output_filename = "output/" + filename
            print(output_filename)
            scale = get_scale(filename)
            cmd = 'ffmpeg -i "' + filename + \
                  '" -y -map v:0 -c:v libx264 -crf 18 -pix_fmt yuv420p' \
                  ' -vf scale=' + scale + ' -g 3 -keyint_min 3 -profile:v high "' + \
                  output_filename + '"'

            cmd_queue.put(cmd)

        except Exception as e:
            invalid_name.append(filename)
            print(filename)

    cmd_queue.join()

    print("Total: ", len(lines))
    print(resolution_stat1)
    print(resolution_stat2)
    print("invalid:")
    print(invalid_name)

