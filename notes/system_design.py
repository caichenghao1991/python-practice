'''

    web service use load balancer (software like nginx, dns load balancing among clones of services)
    cache  memory caching
    static assets: Content Delivery Network   distribution network, is a geographically distributed network of proxy
        servers and their data centers. The goal is to provide high availability and performance by distributing the
        service spatially relative to end users.
    distributed file system  (s3)  indexes (compound). replication slave master, read from slave might have delayed
        result. can read from master or cache to get latest result. database sharding (vertical/horizontal sharding) to
        handle writes. Nosql good for large scale (mongodb, dynamodb)

    steps:
    1. clarify the requirements(functional input/output)/goal/constraints(performance, modifiability, availability,
        scalability, reliability), use cases, constraints, assumptions of the system
    2. estimation of RPS(request per second), storage, bandwidth
    3. define data model and data flow between system components
    4. design high-level components then break components into detail design
    5. detailed design, tradeoff analysis

    6. identify bottlenecks (scale system: load balancer, horizontal scaling, caching, database sharding) and find ways
        to mitigate  (ex. single point of failure: replica data, copies of services)

    common components:
        client application(index controller, chunk controller, watcher, local db), gateway service(authentication,
        routing, load balancing, cache, monitor, log),
        session service: store connection information
        processing server (write api, read api, write api async),
        memory cache, worker service, cloud storage (SQL write master-slave, sql
        read replica), NoSQL, Object Store, request message queue, Metadata
        server,  Metadata DB, response queue, notification server

        client -> dns, cdn, load balancer;    load balancer -> web server(read,write api)
        cdn -> object store     analytics -> object store, sql analytics
        read api -> memory cache, sql read replicas
        write api -> sql write master-slave, object store
            write api -> queue  queue -> worker service   worker service -> NoSQL, Object Store
        crawler service -> NoSQL, queue   queue/query api -> reverse index service(reverse key/value), document service

    Performance (speed for single user) vs scalability (speed in heavy load)
        Latency (time to produce result) vs throughput (results produced in unit time)   generally max throughput under
            acceptable latency
        Availability (always receive result but not always most recent)  vs consistency (read most recent write)
            Partition Tolerance: continue operate with partition/any information failure or missing
            network not reliable either AP(allow for eventual consistency) or CP(atomic reads and writes)

        Consistency: weak: After a write, reads may or may not see it. ex. VoIP, video chat
            eventual consistency: After a write, reads will eventually see it (milliseconds) ex. DNS and email (highly
                available systems)
            strong consistency: After a write, reads will see it.  ex. RDBMS (transaction)
        high availability:
            active-passive fail-over (master-slave fail-over) passive server take over ip resume service if not receive
                heartbeat from active server(took whole traffic). time depends on in 'hot'/'cold' standby
            active-active fail-over (master-master fail-over) both servers are managing traffic, spreading the load
                between them.
            Fail-over adds more hardware and additional complexity, and might loss data during down time
            replication: Master-slave and master-master replication in database
            Availability in parallel vs in sequence: system components in parallel higher overall availability

    Domain Name System (DNS) translates a domain name such as www.example.com to an IP address.DNS is hierarchical, with
            a few authoritative servers at the top level. Your router or ISP provides information about which DNS
            server(s) to contact when doing a lookup. Lower level DNS servers cache mappings, which could become stale
            due to DNS propagation delays.
        DNS results can also be cached by your browser or OS for a certain period of time, determined by the time to
            live. CloudFlare and Route 53 provide managed DNS services. routing method: Weighted round robin,
            Latency-based, Geolocation-based.
        Disadvantage: Accessing a DNS server introduces a slight delay(although mitigated by caching).DNS server
            management could be complex and is generally managed by governments,ISPs, and large companies. DNS services
            have recently come under DDoS attack

    Content Delivery Network (CDN) is a globally distributed network of proxy servers, serving content from locations
        closer to the user. Generally, static files such as HTML/CSS/JS, photos, and videos. The site's DNS resolution
        will tell clients which server to contact
        Serving content from CDNs can significantly improve performance: Users receive content from data centers close
        to them, Your servers do not have to serve requests that the CDN fulfills
        ex. store image as blob, table only have image_id and url, redirect to place in CDN. not save images as file in
            database
        Push CDNs receive new content whenever changes occur on your server.You take full responsibility for providing
            content, uploading directly to the CDN and rewriting URLs to point to the CDN.
        Pull CDNs grab new content from your server when the first user requests the content. You leave the content on
            your server and rewrite URLs to point to the CDN. This results in a slower request until the content is
            cached on the CDN.A time-to-live (TTL) determines how long content is cached. Pull CDNs minimize storage
            space on the CDN, but can create redundant traffic if files expire and are pulled before they have actually
             changed.
        Disadvantage: CDN costs could be significant depending on traffic, Content might be stale if it is updated
            before the TTL expires it. CDNs require changing URLs for static content to point to the CDN.

    Load balancers distribute incoming client requests to computing resources such as application servers and databases.
            In each case, the load balancer returns the response from the computing resource to the appropriate client.
            prevent requests going to unhealthy servers, overloading resources, eliminate a single point of failure, SSL
            termination: Decrypt incoming requests and encrypt server responses so backend servers do not have to
            perform these potentially expensive operations, Session persistence: Issue cookies and route a specific
            client's requests to same instance if the web apps do not keep track of sessions.
        Load balancers can be implemented with hardware (expensive) or with software such as HAProxy.
            Elastic load balancer from AWS
        To protect failures, set up multiple load balancers: active-passive/ active-active
        route traffic based on various metrics: Random, Least loaded, Session/cookies, Round robin or weighted round
            robin, Layer 4 (look at info at the transport layer:source, destination IP addresses, and ports in the
            header to distribute requests), Layer 7 (look at the application layer: header, message, and cookies to
            distribute requests) more flexible but cost more time and computing resources

        Horizontal Scaling: Load balancers can also help with horizontal scaling, improving performance and availability
            Scaling out using commodity machines, instead of Vertical Scaling: upgrade single server on more expensive
            hardware.
            Disadvantages: Scaling horizontally introduces complexity and involves cloning servers:Servers should be
                stateless: not contain any user-related data (sessions, personal data). Sessions can be stored in a
                centralized data store such as a database (SQL, NoSQL) or a persistent cache (Redis, Memcached)
                Downstream servers such as caches and databases need to handle more simultaneous connections
        load balancer is useful when you have multiple servers. route traffic to set of servers with same function.
        Disadvantage: The load balancer can become a performance bottleneck if not enough resources or not configured
            properly. help eliminate a single point of failure results in increased complexity.A single load balancer is
            a single point of failure, configuring multiple load balancers further increases complexity

    Reverse proxy (web server):A reverse proxy is a web server that centralizes internal services and provides unified
        interfaces to the public. Requests from clients are forwarded to a server that can fulfill it before the reverse
        proxy returns the server's response to the client.
        Increased security (hide backend server info, blacklist IPs, limit connection per client), Increased scalability
        and flexibility (Clients only see the reverse proxy's IP, allow scale servers or change their configuration)
        SSL termination, Compression(Compress server responses), Caching(return response for cached requests), Static
        content (serve directly)
        Reverse proxies can be useful even with just one web server or application server
        Disadvantage: increased complexity, A single reverse proxy is a single point of failure, configuring multiple
            reverse proxies (ie a failover) further increases complexity.
        NGINX and HAProxy can support both layer 7 reverse proxying and load balancing.

    Separating out the web layer from the application layerallows you to scale and configure both layers independently.
    Microservices: a suite of independently deployable, small, modular services. Each service runs a unique process and
        communicates through a well-defined, lightweight mechanism to serve a business goal.
        Service Discovery: Systems such as Consul, Etcd, and Zookeeper can help services find each other by keeping
            track of registered names, addresses, and ports. Health checks help verify service integrity and are often
            done using an HTTP endpoint.
        Disadvantage: Adding an application layer with loosely coupled services requires a different approach from an
            architectural, operations, and process viewpoint. add complexity in terms of deployments and operations.

    Relational database management system (RDBMS)
        relational database is a collection of data items organized in tables. ACID is a set of properties of relational
        database transactions.
        Scaling:
            master-slave replication: master serves reads and writes, replicating writes to one or more slaves,which
                serve only reads. Slaves can also replicate to additional slaves in a tree-like fashion. If the master
                goes offline, the system can continue to operate in read-only mode until a slave is promoted to a a
                master or new master is provisioned.
                Disadvantage: Additional logic is needed to promote a slave to a master. potential for loss of data if
                    the master fails before any newly written data can be replicated to other nodes. Writes are replayed
                    to the read replicas. The more read slaves, the more you have to replicate, which leads to greater
                    replication lag. Replication adds more hardware and additional complexity.
            master-master replication: Both masters serve reads and writes and coordinate with each other on writes. If
                either master goes down, the system can continue to operate with both reads and writes.
                Disadvantage: need a load balancer or make changes to application logic to determine where to write.
                    most systems either loosely consistent (violating ACID) or have increased write latency due to
                    synchronization. Conflict resolution comes more into play as more write nodes are added and as
                    latency increases.
            federation (or functional partitioning) splits up databases by function.less read and write traffic to each
                database and therefore less replication lag. Smaller databases result in more data that can fit in
                memory, which in turn results in more cache hits due to improved cache locality. With no single central
                master serializing writes you can write in parallel, increasing throughput.
                Disadvantage: not effective if schema requires huge functions or tables. need to update your application
                    logic to determine which database to read and write. Joining data from two databases is more complex
                    with a server link. adds more hardware and additional complexity.
            sharding: distributes data across different databases such that each database can only manage a subset of
                the data. Ex. shard a user table to tables based on first letter. similar benefit as federation
                Disadvantage: need update your application logic to work with shards, which could result in complex SQL
                    queries. Data distribution can become unbalanced between shards. Joining data from multiple shards
                    is more complex. adds more hardware and additional complexity.
            denormalization: improve read performance at the expense of some write performance. Redundant copies of the
                data are written in multiple tables to avoid expensive joins. might circumvent the need for such complex
                joins for federation and sharding
                Disadvantage: Data is duplicated. Constraints can help redundant copies of information stay in sync,
                    which increases complexity of the database design. not perform well under heavy write load
            SQL tuning: important to benchmark (simulate high-load situations with tools such as ab) and profile (enable
                tools such as the slow query log to help track performance issues) to simulate and uncover bottlenecks.
                schema: use CHAR instead of VARCHAR for fixed-length fields (faster random access). Use TEXT for large
                    blocks of text (allows for boolean searches, store pointer to locate text). Use INT for larger
                    numbers up to 2^32 (4 billion). Use DECIMAL for currency to avoid floating point representation
                    errors. Avoid storing large BLOBS, store the location for the object. use NOT NULL constraint
                    (improve search performance)
                indices: indicies are self-balancing B-tree that keeps data sorted and allows searches, sequential
                    access, insertions, and deletions in logarithmic time. column (SELECT, GROUP BY, ORDER BY, JOIN)
                    could be faster with indices. index can keep the data in memory, requiring more space. Writes slower
                    since the index also needs to be updated.
                Avoid expensive joins: denormalize where performance demands it.
                Partition tables: break up a table by putting hot spots in a separate table to help keep it in memory.
                Tune the query cache: sometimes query cache could lead to performance issues.
        reasons to use SQL: structured data, strict schema, relational data, need for complex joins, transactions, clear
            patterns for scaling, more established: developers, community, code, tools, etc, fast lookups by index
            mutability of data (able to modify), access control
    NoSQL
        NoSQL is a collection of data items represented in a key-value store, document store, wide column store, or a
        graph database. Data is denormalized, and joins are generally done in the application code.  Most NoSQL stores
        lack true ACID transactions and favor eventual consistency. BASE used to describe the properties of NoSQL.

        Key-value store (hash table): A key-value store generally allows for O(1) reads and writes and is
            often backed by memory or SSD. Data stores can maintain keys in lexicographic order, allowing efficient
            retrieval of key ranges. often used for simple data models or for rapidly-changing data, such as an
            in-memory cache layer. Has limited set of operations, complexity is shifted to the application layer
        Document store (key-value store with documents stored as values): stores all information for a given object
            (XML, JSON, binary, etc). provide APIs or a query language to query based on document internal structure.
            provide high flexibility, often used for working with occasionally changing data.
            DynamoDB supports both key-values and documents.
        Wide column store (nested map ColumnFamily<RowKey, Columns<ColKey, Value, Timestamp>>): basic unit of data is a
            column (name/value pair) ex. state: IL. A column can be grouped in column families (analogous to a SQL
            table) ex. address consist city, state, street. Super column families further group column families,
            ex. companies has address, website. You can access each column independently with a row key, and columns
            with the same row key form a row. Each value has a timestamp for versioning and for conflict resolution.
            maintain keys in lexicographic order, allowing efficient retrieval of selective key ranges. Wide column
            stores offer high availability and high scalability. They are often used for very large data sets.
        Graph database(graph): each node is a record and each arc is a relationship between two nodes. Graph databases
            are optimized to represent complex relationships with many foreign keys or many-to-many relationships.
            offer high performance for data models with complex relationships, not yet widely used
        reasons to use NoSQL: semi-structured data, dynamic or flexible schema, non-relational data, no need for complex
            joins, store huge data, very data intensive workload, high throughput for IOPS
            ex. Rapid ingest data, temporary data, frequently accessed tables (Metadata/lookup tables)

    Cache
        Caching improves page load times and can reduce the load on your servers and databases. usually dispatcher check
        client's request previous result in cache first, before actual execution. If not in cache, result a cache miss
        and lookup in database save entry in cache and return.
        Putting a cache in front of a database can help absorb uneven loads and spikes in traffic.
        Client caching: located on the client side (OS or browser), server side, or in a distinct cache layer.
        CDN caching
        Web server caching: reverse proxies and caches such as Varnish can serve static and dynamic content directly.
            Web servers can also cache requests, returning responses without having to contact application servers.
        Database caching: database usually includes some level of caching in a default configuration, optimized for a
            generic use case.
        Application caching: In-memory caches such as Memcached and Redis are key-value stores between your application
            and your data storage.  cache invalidation algorithms such as least recently used (LRU) can help invalidate
            'cold' entries and keep 'hot' data in RAM.
        usually cache database queries and objects (row level query-level, fully-formed serializable objects,
            fully-rendered HTML). avoid file-based caching (difficult cloning and auto-scaling)
        query level cache: hash query as key store result in cache, hard to delete cached result for complex queries,
            change in data need delete all queries involved with the change
        object level cache: assemble the dataset from the database into a class instance or a data structure. Allows for
            asynchronous processing ex. user session, rendered web pages

        When to update the cache
            cache-aside: The application is responsible for reading and writing from storage. The cache does not
                interact with storage directly. Look for entry in cache, resulting in a cache miss. Load entry from the
                database and add entry to cache, return entry. Memcached is an example. Subsequent reads of data added
                to cache are fast. Cache-aside is also referred to as lazy loading. Only requested data is cached
                Disadvantage: cache miss results in three trips, cause a noticeable delay. Data can become stale
                    (outdated) if it is updated in the database. mitigated by setting a time-to-live (TTL) or by using
                    write-through. When a node fails, it is replaced by a new, empty node, increasing latency.
            Write-through: The application uses the cache as the main data store, reading and writing data to it, while
                the cache is responsible for reading and writing to the database
                Disadvantage: When a new node is created due to failure or scaling, the new node will not cache entries
                    until the entry is updated in the database. Cache-aside in conjunction with write through can
                    mitigate this issue. Most data written might never be read, which can be minimized with a TTL.
            Write-behind(back): Add/update entry in cache, cache asynchronously write entry to the data store, improving
                write performance.
                Disadvantage: data loss if the cache goes down prior to its contents hitting the data store. more
                    complex to implement
            Refresh-ahead: automatically refresh any recently accessed cache entry prior to its expiration.reduced
                latency vs read-through if the cache can accurately predict which items will be needed in the future.
                Disadvantage: incorrect prediction can result in reduced performance
        Disadvantage(cache): Need to maintain consistency between caches and the source of truth through complex cache
            invalidation. Need to make application changes such as adding Redis or memcached.

    Asynchronous:
        Asynchronous workflows help reduce request times for expensive operations that would otherwise be performed
        in-line. They can also help by doing time-consuming work in advance, such as periodic aggregation of data.
        Message queues: message queues receive, hold, and deliver messages. An application publishes a job to the queue,
            then notifies the user of job status. A worker picks up the job from the queue, processes it, then signals
            the job is complete. The user is not blocked and the job is processed in the background. during this time,
            optionally can jump to success/main page and make it seems finished already.

            data distribution pattern
                fan-out is a messaging pattern: delivery of a message to one or multiple destinations possibly in
                parallel, not halting the process that executes the messaging to wait for any response to that message.
                Publish–subscribe connects a set of publishers to a set of subscribers.
                Multicast: send message no guarantee delivered
            Redis is useful as a simple message broker but messages can be lost.
            RabbitMQ is popular but requires you to adapt to the 'AMQP' protocol and manage your own nodes.
            Amazon SQS is hosted but can have high latency and has the possibility of messages being delivered twice.
        Task queues: receive tasks and their related data, runs them, then delivers their results. They can support
            scheduling and can be used to run computationally-intensive jobs in the background.
            Celery has support for scheduling and primarily has python support.

        Back pressure: when queue size grow larger than memory, resulting in cache misses, disk reads, and even slower
            performance. Back pressure can help by limiting the queue size, thereby maintaining a high throughput rate
            HTTP 503 status code to try again later.
        Disadvantage: inexpensive calculations and realtime workflows might be better suited for synchronous operations,
            as introducing queues can add delays and complexity.

    Communication
        OSI(open systems interconnection) 7 layers model:
            1. physical (physical structure: cables, hubs). use Hub
            2. data link: transfer data frames('envelopes', contains mac address) between nodes. use switch bridge WAP
            3. network: decide data physical path taken. Packets('letter' contains IP address). use Routers
            4. Transport: ensure correct deliver message in sequence. TCP host to host flow control. TCP, UDP
            5. Session: allow session between processes on different station. Synch & send to ports. use logical ports
            6. Presentation: formats(translate) data to present.  syntax layer, encrypt & decrypt. JPEG/ASCII/TIFF/GIF
            7. Application: serve as window for user. end user layer. application ex.SMTP(Simple Mail Transfer Protocol)

        HTTP: HTTP is a method for encoding and transporting data between a client and a server. It is a request/
            response protocol. self-contained, allowing requests and responses to flow through many intermediate routers
            and servers that perform load balancing, caching, encryption, and compression.
            HTTP request consists of a verb (method: get, post,put,patch,delete) and a resource (endpoint)

        Transmission control protocol (TCP): TCP is a connection-oriented protocol over an IP network. Connection is
            established and terminated using a handshake. All packets sent are guaranteed to reach the destination in
            the original order and without corruption through: sequence numbers and checksum fields for each packet,
            acknowledgement packets and automatic retransmission. If the sender does not receive a correct response, it
            will resend the packets. If there are multiple timeouts, the connection is dropped. TCP also implements
            flow control and congestion control. To ensure high throughput, web servers can keep a large number of TCP
            connections open, resulting in high memory usage. connection pool can help or use UDP in suitable case
            TCP is useful for applications that require high reliability but are less time critical.

        User datagram protocol (UDP): UDP is connectionless. Datagrams (analogous to packets) are guaranteed only at the
            datagram level. Datagrams might reach their destination out of order or not at all. UDP does not support
            congestion control. UDP can broadcast, sending datagrams to all devices on the subnet. This is useful with
            DHCP because the client has not yet received an IP address

        XMPP(Extensible Messaging and Presence Protocol): open communication protocol designed for instant messaging
            (IM), presence information, and contact list maintenance.[2] Based on XML (Extensible Markup Language), it
            enables the near-real-time exchange of structured data between two or more network entities.

        Remote procedure call (RPC): a client causes a procedure to execute on a different address space, usually a
            remote server. The procedure is coded as if it were a local procedure call, abstracting away the details of
            how to communicate with the server from the client program. Popular RPC frameworks include Protobuf, Thrift,
            and Avro.
            RPC is a request-response protocol:
                Client program: Calls the client stub procedure. The parameters are pushed onto the stack like a local
                    procedure call.
                Client stub procedure - Marshals (packs) procedure id and arguments into a request message.
                Client communication module - OS sends the message from the client to the server.
                Server communication module - OS passes the incoming packets to the server stub procedure.
                Server stub procedure - Unmarshalls the results, calls the server procedure matching the procedure id
                    and passes the given arguments.
                The server response repeats the steps above in reverse order.
            samples RPC call: GET /getStudent?data=anId   POST /create student {"data":"anId"; "data2": "another value"}
                RPC is focused on exposing behaviors. RPCs are often used for performance reasons with internal
                communications, as you can hand-craft native calls to better fit your use cases.
            Disadvantage: RPC clients become tightly coupled to the service implementation. new API must be defined for
                every new operation or use case. difficult to debug. not be able to leverage existing technologies out
                of the box.

        Representational state transfer (REST): REST is an architectural style enforcing a client/server model where the
            client acts on a set of resources managed by the server. The server provides a representation of resources
            and actions that can either manipulate or get a new representation of resources. All communication must be
            stateless and cacheable.
            Qualities: Identify resources (URI in HTTP): use the same URI regardless of any operation.
                Change with representations (Verbs in HTTP): use verbs, headers, and body.
                Self-descriptive error message (status response in HTTP): Use status codes, don't reinvent the wheel.
                HATEOAS (HTML interface for HTTP): web service should be fully accessible in a browser.
            sample REST: GET /student/anId     PUT /student/anId {"data": "another value"}
            REST is focused on exposing data. It minimizes the coupling between client/server and is often used for
                public HTTP APIs. REST uses a more generic and uniform method of exposing resources through URIs,
                representation through headers, and actions through verbs. Being stateless, REST is great for horizontal
                scaling and partitioning.
            Disadvantage: not good if resources are not naturally organized or accessed in a simple hierarchy. typically
                relies on a few verbs (GET, POST, PUT, DELETE, and PATCH) which sometimes doesn't fit your use case.
                Fetching complicated resources with nested hierarchies requires multiple round trips between the client
                and server to render single views. Over time, more fields might be added to an API response which is not
                needed by older clients, cause larger payload and latencies.

    Security
        Encrypt in transit and at rest.
        Sanitize all user inputs or any input parameters exposed to user to prevent XSS and SQL injection.
        Use parameterized queries to prevent SQL injection.
        Use the principle of least privilege.



























'''