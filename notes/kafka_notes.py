'''
    kafka is just like a messaging system mostly for event driven system. It is a distributed platform/application. In
        production environment, there are multiple kafka servers (broker) as kafka cluster. There can be multiple
        message producers, and consumers. kafka act as a queue can store and retrieve message. It is fault-tolerant,
        messages are replicated in multiple brokers. It is scalable by adding more brokers, consumers, and producers at
        any time.

    each kafka server can have multiple topics, each topic can have many partitions. producers send message to
        different partition. consumers inside consumer groups consume the messages in the topic.

    zookeeper is a distributed, open source configuration, synchronization service for managing configuration (which
        messages consumer has read, cluster information, topic configuration), any configuration updates will send to
        zookeeper and notify the downstream


    start zookeeper and kafka
        download from https://kafka.apache.org/downloads
        update /config/zookeeper.properties and  /config/server.properties   for changing the  dataDir and log.dirs
            server.properties: advertised.listeners=PLAINTEXT://your.host.name:9092  # kafka server ip port 
                zookeeper.connect=localhost:2181   # connect to ip:port zookeeper
            zookeeper.properties: log.dirs=D:/kafka_2.12-3.0.0/data/kafka
        start zookeeper and kafka with:
            zookeeper-server-start.sh ../config/zookeeper.properties
                .\bin\windows\zookeeper-server-start.bat .\config\zookeeper.properties
            JMX_PORT=8004 kafka-server-start.sh (.bat) ../config/server.properties   # jmx port for kafka manager
                .\bin\windows\kafka-server-start.bat .\config\server.properties
    kafka manager
        graphic interface to manage kafka cluster. easier to grasp, no need remember all the sh command inside bin
            folder.
            need install java 11+.  clone https://github.com/yahoo/CMAK
            ./sbt clean dist    go inside /target/universal  unzip cmak-3.0.0.5.zip
            /target/universal/cmaak-3.0.0.5/conf/application.conf
                cmak.zkhosts="localhost:2181"
            start zookeeper, kafka, then kafka manager
            bin/cmak -D config.file=conf/application.conf -Dhttp.port=8080
        localhost:8080 manager page
            add cluster: cluster name, zookeeper host, enable JMX polling, poll consumer information
            generate __consumer_offsets by default  (keep track of partition id consumer finished with)
            add topic: topic name, partitions, replication factor


    kafka topic
        producers connect and publish message to the kafka topic and partition they chose. multiple producers and
            consumers can connect to one topic. topic can be considered as a logical entity, messages are stored inside
            topic partitions. each kafka servers inside cluster have topics.
        partition
            Kafka topic is divided into multiple partitions.
            Partitions can be considered as a linear data structure, like array, named commits logs.
            each partition has a partition number, and each element inside partition has increasing index called offset
            Data are pushed at the end of partition and is immutable after publish
            Partitions for a topic are distributed across the whole cluster

        single producer, 1 server 1 partition in 1 topic, publish all message to that partition. 1 consumer consume all.
        single producer, 1 server 2 partition in 1 topic, publish all message to 2 partitions randomly if not specified.
            1 consumer will consume the partition in round-robin manner
        1 server 1 partition 2 consumers. same partition can't be assigned to multiple consumer in same group, only 1 '
            consumer will get the message (whoever come first). one message can be consumed multiple times for consumers
            in different group
        1 server 2 partitions 2 consumers from same consumer group. each consumer will randomly assigned to a partition,
            and get message send to that partition only.

        producer
            need configuration for bootstrap kafka server, topic name, value_serializer (can't send object)
            pip install kafka-python  faker
            data.py
                from faker import Faker
                faker = Faker()
                def get_user():
                    return {
                        "name": fake.name(),
                        "address": fake.address(),
                        "created_at": fake.year()
                    }
            producer.py
                from kafka import KafkaProducer
                import json
                def json_serializer(data):
                    return json.dumps(data).encode("utf-8")
                def get_partition(key, all, available):   # key byte, all_partition, available_partition
                    return 0    # always send to partition 0

                producer = KafkaProducer(bootstrap_servers=['192.168.0.10:9092'], value_serializer=json_serializer)
                    # specify kafka server ip port, serializer
                    producer = KafkaProducer(bootstrap_servers=['192.168.0.10:9092'], value_serializer=json_serializer,
                        partitioner=get_partition)   # specify producer to send to a specified partition
                while True:
                    user = get_user()
                    producer.send("topic_user", user)  # send data to exist topic name in kafka server
                    time.sleep(3)

        consumer
            consume the message from kafka topic (partition). Every consumer is assigned to a consumer group. If no
                group_id is provided, a random group_id will be created and assign to the consumer.
                consumer need topic name, bootstrap_server (kafka), and group_id to consume a message

            consumer.py
                consumer = KafkaConsumer("topic_user",bootstrap_servers=['192.168.0.10:9092'],
                    auto_offset_reset='earliest', group_id="consumer-group-a")
                    # topic name, kafka server, consume from starting or latest, group id
                for msg in consumer:
                    print("Message: {}".format(json.loads(msg.value))

                terminal: python consumer.py  # daemon process, will keep listen to the topic

        consumer group
            logical grouping of one or more consumer, mandatory for consumer to register to a consumer group. consumer
                instances are separate process. consumer instance of same consumer group can be on different node (kafka
                server)


    replication in kafka
        kafka is fault-tolerant (continue operating without interruption with one or more its component fail)
        each partition is replicated across multiple server, only one partition at a time within duplication become
        leader, others are followers. leader handles all read and write request for that partition, and follower
        passively replicate the partition of leader. zookeeper will know which is leader and which is follower. If
        leader went down, zookeeper will start polling for new leader

        factor.json
        {"version":1,
            "partitions":[
                {"topic":"signals","partition":0,"replicas":[0,1,2]},
                {"topic":"signals","partition":1,"replicas":[0,1,2]},
                {"topic":"signals","partition":2,"replicas":[0,1,2]}
        ]}
        kafka-reassign-partitions --zookeeper localhost:2181 --reassignment-json-file factor.json --execute



'''