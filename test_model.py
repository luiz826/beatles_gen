import gpt_2_simple as gpt2 

sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess)

gpt2.generate(sess)

print(gpt2.generate(sess, length=100, prefix="Lucy in", temperature=1.0))