require('dotenv').config();
const amqp = require('amqplib/callback_api');

const url = process.env.RABBITMQ_URL;
const queue = process.env.QUEUE_NAME;

amqp.connect(url, function (error0, connection) {
  if (error0) {
    throw error0;
  }

  connection.createChannel(function (error1, channel) {
    if (error1) {
      throw error1;
    }

    channel.assertQueue(queue, {
      durable: false,
    });

    console.log('Waiting for messages in %s', queue);

    channel.consume(
      queue,
      function (msg) {
        console.log('Received: %s', msg.content.toString());
      },
      {
        noAck: true,
      }
    );
  });
});
