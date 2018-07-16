import data
import model
import tensorflow as tf
import os

main_dir = os.path.dirname(os.path.abspath(__file__))
run_name = "Solanaceae-ZX"

d = data.Data()
v = data.Validator()
m = model.Model(d.smiles_vocabulary)

batch_size = 120
validation_batch_size = 20

def saveModel (session, directory):
    saver = tf.train.Saver()
    saver.save(session, main_dir + "/" + directory)

# Might have to use SGD
with tf.Session() as session:

    session.run(tf.global_variables_initializer())

    validation_loss = tf.Variable(0, dtype=tf.float32, trainable=False)

    tf.summary.scalar("Loss", m.loss, family="Loss")
    tf.summary.scalar("Validation-loss", validation_loss, family="Loss")

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(main_dir + "/tensorboard/" + run_name, session.graph)

    while True:

        # Create validation
        smiles, labels, weights = v.getValidationSet(amount=validation_batch_size)
        session.run(tf.assign(validation_loss, m.loss), feed_dict={m.input: smiles, m.expect: labels, m.batch_size: validation_batch_size, m.loss_weights: weights})

        smiles, labels, weights, smiles_raw = d.getData(amount=batch_size)
        summary, iteration, output, loss, _ = session.run([merged, m.global_step, m.output, m.loss, m.optimiser], feed_dict={m.input: smiles, m.expect: labels, m.batch_size: batch_size, m.loss_weights: weights})

        print("Iteration {}, loss: {}".format(iteration, loss))
        print("E: {}, O: {}".format(labels, output))

        writer.add_summary(summary, iteration)

        if iteration % 200 == 0:
            saveModel(session, "saves/" + run_name)
            print("Saved.")
