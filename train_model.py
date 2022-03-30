import gpt_2_simple as gpt2

file_name = "./data/beatles.txt"
model_name = "124M"
gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/124M/

sess = gpt2.start_tf_sess()
gpt2.finetune(sess,
              file_name,
              model_name=model_name,
              steps=5000)   
gpt2.generate(sess)

