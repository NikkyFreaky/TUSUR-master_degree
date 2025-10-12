require('dotenv').config();
const amqp = require('amqplib/callback_api');

const url = process.env.RABBITMQ_URL;

amqp.connect(url, function (error0, connection) {
  if (error0) {
    throw error0;
  }

  connection.createChannel(function (error1, channel) {
    if (error1) {
      throw error1;
    }

    var queue = process.env.QUEUE_NAME;
    var msg = 'Hello from service 1';

    channel.assertQueue(queue, {
      durable: false,
    });
    channel.sendToQueue(queue, Buffer.from(msg));

    console.log(' [x] Sent %s', msg);

    setTimeout(function () {
      connection.close();
      process.exit(0);
    }, 500);
  });
});
