import subprocess

# Initialize steps
# steps = 384000


# Loop condition: steps greater than or equal to 50000

# 定义seg_list
seg_list = [
    "Forehead", "Eyes",  "right_eye", "left_eye","Eyelids", "Eyelashes", "Eyebrows",
    "Nose", "Nostrils", "Cheeks", "Mouth", "Lips", "Teeth", "Tongue",
    "Chin", "Jawline", "Ears", "Hair", "Neck",
    # "Skin texture", "Wrinkles",
    # "Scars"
]

# 遍历seg_list并处理每个元素
for prompt in seg_list:


    print("=============================================== processing ： ", prompt, "==============")
    subprocess.run([
        'python', 'img2mask.py',
        '--prompt', prompt,
        '--image_path',
        '/public/hezhenliang/users/gaoge/VIPLFaceMDT/LD-ZNet/testing_scripts/examples/FFHQ_256_sample_384000_8.jpg',
    ])