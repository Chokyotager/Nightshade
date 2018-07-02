import data
import model
import tensorflow as tf

d = data.Data()

m = model.Model(d.smiles_vocabulary)

batch_size = 1

# Might have to use SGD
with tf.Session() as session:

    session.run(tf.global_variables_initializer())

    while True:

        smiles, labels, smiles_raw = d.getData(amount=batch_size)
        iteration, output, loss, _ = session.run([m.global_step, m.output, m.loss, m.optimiser], feed_dict={m.input: smiles, m.expect: labels, m.batch_size: batch_size})

        print("Iteration {}, loss: {}".format(iteration, loss))
        print("E: {}, O: {}".format(labels, output))
