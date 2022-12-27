'''
    docker
        VM: infrastructure, Hosting Operating system, Hypervisor, Guest OS, Bins/Libs, Apps
        Docker: infrastructure, Hosting Operating system, Docker Engine, Bins/Libs, Apps
            fast, small, can runs many docker
        docker image can't be changed unless create new one. running multiprocess program not recommended
        based on container technology (virtualization), contain docker hub, image, container. docker.io default has two
        process (server(daemon), client). one image can run multiple times, create many containers (child process)
        complicated image build on simpler image

        ali cloud and tencent cloud accelerator for chinese docker image

        apt install docker.io  # unbuntu
        apt install openssh-server      sudo service ssh start
        service docker status   # check status of docker (run/stop)
        # docker images command
        docker search redis     # search image name in docker hub
        docker pull redis       # install environment for image
        docker images           # check all local images    (repository, tag, image id, created, size)
        docker commit -m="has update" -a="runoob" xxx redis:v2   # update image,  param: message, author, container ID,
                                # updated image name to docker hub
        docker rmi redis1      # remove image by id /image name
        docker build -t redis1:v2 .   # build image . is context path for docker file
                                # -t specify image name
            # docker build github.com/crack/docker-firefox   #  build image from internet url
        docker tag xxxx redis1:v2         # specify image id change name to image name


        # docker container command

        docker run --name redis1 redis        # specifies container name, image name
            # run on current terminal window in sync mode
            # docker run ubuntu:latest /bin/echo helloworld
            # add additional command to override default command after start container
            # docker run --name -v /data/redis:/data redis   # map container data to host machine, allow access data in
                # new container after the old container is destroyed
                # docker volume create /data/redis    # create shared volume for docker container, won't get destroyed
                    # after container destroyed
                # /data/redis  host directory,  /data  container directory
            # -rm  remove container automatically after finish running job
            docker run -dit --name redis1 -p 6378:6379 redis      # -d:backend, -i:can enter container, -t:can open
                 terminal, 6378 ubuntu machine port(place running docker commands),  6379 container port  run in backend
        docker inspect containerid    # check container info
        docker ps a            # show all container (stopped as well, check for container id)
                u: show program based on user     # x show all program      # ps aux  # can write together
        docker start redis1
        docker stop redis1
        docker rm redis1        # remove container   -f force remove
        docker exec redis1 ls   # execute command inside container
        docker exec -it redis1 /bin/sh    # enter container at specific path
            exit()             # leave container
        docker export containerID > redis1.tar    # export container snapshot to local pc
        $ cat redis1.tar | docker import - redis1:v2  # create image with local snapshot
            or  docker save -o ~/redis.tar redis1  # save imageID/name snapshot
                docker load --input ~/redis.tar   or   docker load < ~/redis.tar
        docker logs   # check logs
        docker top containerid   # check container status

        Dockerfile
            FROM: ubuntu-dev:latest   # base:image version
            MAINTAINER cai caihogwarts@gmail.com      # declare author
            USER <username>[:<group>]   # specify user to run later commands (user and group must exist)
                USER user:group   uid:gid
            EXPOSE <port1> [<port2>]   # specify exposed port   docker run -P will random choose one EXPOSE port
            LABEL <key>=<value> <key>=<value> <key>=<value> ...  # add meta data to image
            ENV <key1>=<value1> <key2>=<value2>...    # setting environment variable
                # ENV NODE_VERSION 7.2.0
            ARG <key1>=<value1> <key2>=<value2>...    # environment variable only valid during building phase
            VOLUME <path>             # default anonymous path, save data to local pc path
            WORKDIR /usr/src          # change work directory
            COPY [--chown=<user>:<group>] <path_pc>...  <path_cont>   # change owner and group of copied file (optional)
                # path_pc: filename and path of context running the container   COPY hom* /mydir/
                # path_cont: path to copy the files inside container (create path automatically if not exist)
                # COPY . .    # copy all current directory files to WORKDIR
            RUN apt update            # execute terminal (shell) command when docker container is building
                RUN ['apt update','param1','param2']     # execute command execute format
                RUN yum -y install wget     RUN tar -xvf redis.tar.gz
            HEATHCHECK [option] CMD <command>            # set command or program to monitor docker container status
                HEATHCHECK NONE                          # block base healthcheck inside base image
            ENTRYPOINT ["<executeable>","<param1>","<param2>",...]: similar to CMD, but provide fix param won't be
                changed by docker run param. docker run can add --entrypoint to change command to run during runtime
                only last ENTRYPOINT will be executed
                ENTRYPOINT <shell command>
                ENTRYPOINT ["nginx"]    ENTRYPOINT [ "curl", "-s", "http://ip.cn" ]
            CMD python3               # command to execute to run other program, when docker container is running
                                      # param can be override by docker run
                CMD <shell command>
                CMD ["<executable file or command >","<param1>","<param2>",...]   # recommended way
                CMD ["<param1>","<param2>",...]  # provide default param to ENTRYPOINT
                    # only last CMD will be executed, previous one inside same docker file will be ignored
                    CMD /bin/sh -c 'nginx -g "daemon off;"'    CMD ["-g","daemon off;"]   CMD ["/usr/bin/wc","--help"]
                    # docker run -it -p 8080:8000 <docker_image> ls -l    command with param (ls -l) will override
                        cmd command, need entrypoint ahead of cmd in this case to avoid must execute command overridden
                CMD设置的指令在镜像运行时自动运行，无法追加指令，只能把指令全部覆盖
                ENTRYPOINT设置的指令在镜像运行时与CMD一样，可以在新建镜像时设置的指令后追加新的指令，也可以使用 --entrypoint 覆盖指令。
            ONBUILD <command>         # command won't be executed for building this image, only run command when some
                                      # other image have 'from thisImageName' (use this image as base image)
        ex.
        FROM python:3.8-slim-buster
        WORKDIR /app
        COPY . .
        RUN pip3 install -r requirements.txt
        CMD ["python3", "app.py"]

        docker build -t myapp .   # build image . is context path if needed from pc running the container
        docker run -p 80:5000 -d myapp    # 80 is pc port, 5000 is container port   -d rin in background


        Docker Compose
            docker compose is a tool used for define and run application on multiple docker container. use YML file to
                configure all the application's service (version, services(ports, build, volume, links, args, target,
                dockerfile, context, command, container_name, depends_on, cap_add, cap_drop, cgroup_parent,
                deploy(endpoint_mode, labels, mode, replicas, resources, restart_policy, rollback_config, update_config)
                , devices, dns, dns_search, entrypoint, env_file, environment, expose, extra_hosts, healthcheck, image,
                logging, network_mode, restart, secrets, security_opt, stop_grace_period, stop_signal, sysctls, tmpfs,
                ulimits, volumes). use docker-compose up to start the application
            docker-compose up   # create and run environment with docker-compose.yml
                # docker-compose -f nginx.yaml up   # if name is not docker-compose.yml, need add -f and specify name
            docker-compose down  # stop and remove container    --volumes   delete volumes

            docker-compose.yml
            version: '2'
            command: [bundle, exec, thin, -p, 3000]  # optional, overwrite default command after container initialized
                # command: bundle exec thin -p 3000
            services:
                ngix-app:   # custom service name
                    build: .                        # must have build or image
                        context: ./dir              # specify dockerfile location, can be git url
                        dockerfile: Dockerfile      # specify Dockerfile name, optional if same as default Dockerfile
                        args:
                            - buildno=1
                            - password=secret
                    image: dockercloud/hello-world  # or use image id instead of build
                    depends_on:                     # first initialize depended service then current service
                        - mysql
                    links:                          # link to other service
                        - db
                        - redis
                    container_name: app             # optional, specify container name
                    ports:                          # access port
                        - 8080
                        - 80:8080                   # pc port: container port
                    dns: 8.8.8.8                    # define single dns
                        #dns:                       # define list dns
                            - 8.8.8.8
                            - 9.9.9.9
                    expose:                         # optional Ports are not exposed to host machines,
                        - "3000"                    # only exposed to other services.
                        - "8000"
                    networks:                       # add service into network
                        - front-tier
                        - back-tier
                    env_file: .env                  # single environment file
                        # env_file:                 # multiple environment file
                            - ./common.env
                            - ./apps/web.env
                    environment:                    # optional add environmental variable in Dockerfile
                        USER: "root"
                        PASSWORD: "123"
                    labels:                         # optional add meta data
                        - "com.example.description=Accounting webapp"
                    volumes:                        # optional add a file directory
                        - /var/lib/mysql:/var/lib/mysql
                    volumes_from:                   # add all directory from other service/container
                        - service_name
                        - container_name
                    entrypoint: /code/entrypoint.sh # optional override entrypoint in Dockerfile
                    extends:                        # extend same base setting from current file service or yml file
                        file: common.yml
                        service: webapp
                    logging:                        # specify logging config
                      driver: syslog
                      options:
                        syslog-address: "tcp://192.168.0.42:123"

                mysql:
                    image: mysql
                    ports:
                        - "7706:3306"
                    environment:
                        MYSQL_ROOT_PASSWORD: "123"
                        MYSQL_DATABASE: "test"
                        MYSQL_USER: "root"
                        MYSQL_PASSWORD: "123"
                    volumes: permanent-shared-volume:/var/lib/mysql
            networks:
                front-tier:
                    driver: bridge          # host, none,  service, container
                back-tier:
                    driver: bridge
            volumes:
                permanent-shared-volume:             # permanent volume, won't destroyed after container gone

        Docker Machine
            install docker on virtual machine. use docker-machine to manage all the docker host
            docker-machine (version, ls, create --driver virtualbox test, ip test, stop test, start test, ssh test,
                active, config, env, inspect, kill, provision, regenerate-certs, restart, rm, scp, mount, status,
                upgrade, url, help) # test is the docker container name

        DockerSwarm
            docker containers cluster management
            manager node:  docker swarm init --advertise-addr 192.168.99.107 # start main node with ip
            workers node: docker swarm join --token xxxx 192.168.99.107:2377  # join the swarm as worker node
            docker info   # check cluster information
            docker service create --replicas 1 --name redis1  # start service on manager node,  name is image name
            docker service ps redis1    # check service status
            docker service scale redis1=2   # change workers number for service
            docker service rm redis1    # remove service
            docker service create --replicas 1 --name redis --update-delay 10s redis:3.0.6   # create and update service
            docker node ls          # check all workers and manager info
            docker node update --availability active[drain] swarm-worker1   # update worker availability to run service


    jenkins
        pipeline: check general, build triggers, project options tabs (pipeline script or jenkins file in project(on
            git)), script use groovy

            def file_name = "Jenkinsfile"   # define variable must ahead of script
            pipeline{
                agent any
                agent{    # run jenkins on specific cluster node
                    node{
                        label 'slave-1'
                    }
                }
                environment {
                    NUGET_KEY = 'abc'   # usage   bash: env.NUGET_KEY   sh: $NUGET_KEY
                    # predefined env variables: BRANCH_NAME, GIT_COMMIT, GIT_PREVIOUS_SUCCESSFUL_COMMIT
                }
                triggers {
                    githubPush()
                }
                stages{
                    stage('Checkout'){
                        steps{
                            echo 'hello'   # add steps using Pipeline Syntax link below
                        }
                    }
                    stage('Build'){
                        steps{
                            git branch:...  # generated by Pipeline Syntax need specify git location during build
                            bat ...   # bash command generated by Pipeline Syntax
                            result = sh (script: "git log -1|grep 'Release'", returnStatus: true)
                        }
                    }
                    stage('Test'){
                        steps{
                            git branch:...  # generated by Pipeline Syntax need specify git location during build
                            bat ...   # bash command generated by Pipeline Syntax
                        }
                    }
                    stage('Deploy'){
                        when {
                            branch "master"   # only run on master branch
                            }
                }
            }

        add credentials:  Jenkins_url/credentials/store/system/domain/_/
            add secret text (define secret, credential_id)         usage:  credentials('credential_id')


    Kubernetes
        kubernetes is a production-grade container orchestration, automated container deployment, scaling and management
            with extra functionality like: load balancing, allocate resource and servers, fail over.
        Pod is the smallest executable unit(single/multiple container, ex. web service server, database server), usually
            one main application per pod, each pod gets its own IP address. It is a layer of abstraction over container.
            Pod IP address from Node's IP range
        Service: A service is responsible for enabling network access to a set of pods. each pod in replica set share a
            same service(a component, not process) with fix IP, port, even the container died won't change the service
            IP associate with this pod, will reassign a replica on different node with same service IP instead of
            assigning a new IP each time container failed. And pods communicate to each other via service. All the
            replicas of application share a service with same service IP. Service also act as a load balancer.
            user can call a single IP (service) instead of calling different individual pod ip.
            service can connect with other cluster component or outside cluster, like browser (through ingress),
            database. service connect to pods via label of pod in same replica set and service selector (match all
            key-value pair). When request come, service will randomly send to one (prefer least load) pod in the replica
            set(same labels as service selector), with the port of pod's targetPort. service span all nodes with replica
            pod inside.
            When create service, kubernetes create an endpoint object with same name of service('kubectl get endpoints')
            use this endpoint object to track which pods are member/endpoint of the service, updated if pods die.
            If service has multiple port corresponding to multiple application (different port), service ports need
            to have a '- name:' to specify the port name
            there are 4 type of services: ClusterIP, Headless, NodePort, LoadBalancer
            ClusterIP: default type of service, if not specify any type.
            Headless: handle client's need communicate with specific pods, usually for stateful application, since pods
            are not identical(master, worker). client need to know IP address for each pod via: API call to kubernetes
            API server (inefficient), or use DNS lookup (in service config file, under spec add clusterIP: None), this
            will return pods' IP addresses instead of service (cluster) IP, and no cluster IP address is assigned in
            this case.
            NodePort: create service in each worker node with a static port(NodePort(range 30000-32767)) and browser can
            directly connect with node ip and node port. compare to default, service IP only accessible within the
            cluster, need ingress to connect to browser. Node port create external port for all nodes in replica set,
            and make nodes publicly accessible to outside, so not efficient and secure. so nodeport usually only for
            testing purpose and not for production. Wither use ingress + clusterIP or load balancer for external access.
            LoadBalancer: service become accessible externally via cloud providers load balancer. This will also create
            node port for the node where pod replica lives in, but node port only accessible via cloud load balancer

        Ingress: The request first goes to ingress, so user can access page via https://my-app.com, then ingress route
            traffic to service IP address (http://ip:port). Can have many ingress to connect internal service to some
            domain name, but only one Ingress Controller Pod required to have to evaluate all the rules and manage
            redirection, also act as entrypoint to cluster, implemented by 3rd-party(ex. K8s Nginx Ingress Controller).
            'minikube addons enable ingress' to start k8s nginx implementation of ingress controller for minikube.
                check via 'kubectl get pod -n kubesystem'  # check nginx-ingress-controller
            Can have cloud loud balancer connect to ingress controller, or configure some entrypoint yourself. Need
            entry point either inside of cluster or outside as separate (proxy) server (software/hardware solution)

        Configure Map: external configuration of application, connect to the pods. so if need to change the config of
            the service, no need to rebuild the image with new application properties file.
        Secret: similar to configure map, (credential, certificate) but not stored in plain text format. secret are
            accessible by pods
        Volume: attach a (local/remote) physical storage to pod. So all the data will be persisted with pod restarts.
            Kubernetes don't manage data persistence, so you have to backup with other solution.
            3 component of kubernetes: persistent volume, persistent volume claim, and storage class
            Kubernetes default don't provide data persistence, data is gone after pod life cycle ends.
            When try to achieve data persistence, storage must be available on all nodes, since not certain pod will be
            restart on which node. Storage must not depend on pod life cycle. Storage need to survive even cluster
            crashes. local volume depends on node and won't survive crash. So for persistence use remote storage

            Persistent Volume: pre-configured directory (session file, configure file), it's a cluster resource to store
                data, created via yaml. It needs actual physical storage (local disk, nfs(Network File Sharing) server
                or cloud storage). its like external plugin to the cluster, each application in the cluster can configure
                to use different storage. kubernetes only provide storage interface, you need to create and manage
                (create backup, check corruption) storage. It is outside namespace, and accessible to whole cluster.
                persistent volume should be there already before pod refer to it.
                ConfigMap and Secret are local volume not created by persistent volume and claim, but managed by
                kubernetes. Also need to mount configMap/secret to volume of pod
            Persistent Volume Claim: application need to claim the persistent volume, created via yaml configure. only
                persistent volume satisfied the claim will be used for application. In the pod configure file need add
                volumes section under spec. Pods request volume through persistent volume claim, trying to find a
                persistent volume that satisfy the claim. pod and it's claim must be in the same name space. After
                find a persistent volume, volume is mounted into the Pod and then mounted into one/ multiple container.
                After container died, new container will have access and able to see the same data on the volume.
                One pod can claim multiple persistent volume.
            Storage Class: provisions persistent volume dynamically, created by yaml config file. each storage has own
                provisioner (internal provisioner with prefix kubernetes.io). Need to specify in persistent volume claim
                config file under spec: add storageClassName: value (same as storage class config meta name)
                When pod claim storage via persistent volume claim,  persistent volume claim will request storage from
                storage class, storage class provision and create the persistent volume that meets the pod's claim.

        Deployment: A deployment is responsible for keeping a set of pods running. blueprint for pods, another layer of
            abstraction over pods. User specify number of replica in deployment. Deployment can only replicate
            application, not database(has state(changed data)) deployment manage all the replica set (replica of pod).
            user only need manage deployment, any thing below is handled by kubernetes
        Stateful set: used to replicate database for pods. make sure synchronized data across replica. stateful set is
            harder than deployments, so it's common to host database outside kubernetes cluster.
            stateful applications: databases, or applications that stores data to keep track of state via storage.
                deployed via Stateful Set. replica pods are not identical, they have pod (sticky) identity. can't be
                created/deleted at same time/ random order. can't randomly addressed. They are created from same
                specification, but not interchangeable. persistent identifier across any rescheduling (replaced pod keep
                original identity)
            stateless applications: don't keep record of state, each request is completely new, sometime forward request
                to stateful application. deployed via deployment. Pods are identical and interchangeable, created in
                random order with random hashes, one service load balances to any pod.
            configure storage the same way. But when scaling the stateful application(database), only master can read
            and write, worker can only read, and each pods have their own storage (replica of storage to access by only
            themself), and need to synchronizing the data. When new worker replica join, it create own storage by
            cloning data from previous pod. When pod died and got replaced, it keep the pod state(stored remotely).
            unlike deployment get deploymentName-randomHash for the pod identity, stateful set has StatefulSetName-$(ordinal)
            next pod is only created if previous is up and running. deletion is in reverse order, start from the last
            one. Each pod in stateful set get own dns endpoint([podname].[governing service domain]) from load balancer
            service. with new pod replaced old one, although IP changes, sticky identity: pod name and DNS endpoint
            stays same. sticky identity retain state and role.
            You still need configure the cloning and data synchronization, make remote storage available, managing and
            backup, kubernetes can't provide help on those, because stateful applications not perfect for containerized
            environments, mostly for stateless application.
        Nodes(worker): have single/multiple pod (must have kubelet(process that schedule the container, interacts with
            node and container, assign resource to container), kube-proxy(intelligently forward request from service to
             pod), Pod(container runtime))
        Master Node: There is one or multiple(for high availability) central control computer(control plane)
            k8s cluster services to manage nodes and pods. It use api to communicate with nodes. It will monitor network
            status of nodes to balance the load. It can also issue command to nodes (replace drained pod with pod in
            replica set)
            master node: api server(user interact with api server, acts as a (authenticated) gateway to update/query
            request for pods state), scheduler(decide on which node new pod should be scheduled), controller manager
            (detect cluster state changes. ex.pods crashed), etcd database(Cluster state store/changes (communication
            between master and nodes) stored in the key value pair)
            master node need less resource since it's nodes that do actually jobs.
            master and nodes recommended located on different machine, can be same machine though
        Cluster: control plane and corresponding nodes
        Namespace: can have multiple namespace(cluster) in cluster to organize resource. default has some namespace:
            kube-node-lease(node associated lease object in namespace, heartbeats of nodes to determine availability);
            kube-public(public accessible data, a config map with cluster information, use 'kubectl cluster info' to get
            print values ); kube-system(don't modified, for system processes(master, kubectl)); default (used to hold
            created new name spaces and other resource you created), kubernetes-dashboard(for minikube)
            use namespace to group resource with same kind together; allow different team to work on same deployment
            within same k8s cluster, using their own namespace(cluster in k8s cluster) will make sure they don't
            override(interrupt) each other deployment, and each team have isolated environment, and can limit resource
            assign to each namespace; can make different environment (prod, dev), blue/green deployment their own
            namespace, so they can retrieved shared other namespace information, without need of create two clusters
            Since can't access most resources from another namespace, each namespace need their own configmap, secret.
                but you can access service in other namespace.
            volume and node can't be assign to namespace. use 'kubectl api-resources --namspaced=false/true' to list
                resource (not) bound to namespace
            by default, components are created in default namespace if not specified. after command add
                'kubectl apply -f deployment.yaml --namespace=my-namespace' to assign to specified namespace
                or inside configuration file(recommended): under metadata add:    namespace: my-namespace
            when using get command need add -n to specify namespace if not default:
                'kubectl get configmap -n my-namespace'
            use kubens to change default namespace to specified namespace a, so no need -n when refer to namespace a
                in command line:  $ brew install kubectx       $kubens
                $ kubens my-namespace   # change default namespace to my-namespace
        Helm: package manager for kubernetes, package yaml files and distribute in repositories. create Helm charts
            and push to Helm repository, so other people can download and use with same need(ex. build nginx cluster)
            helm search <keyword>   or search in Helm Hub.  install with 'helm' install [chartname]'
                'helm install --values=my-values.yaml [chartname]'  will merge values in my-value.yaml and default
                values.yaml values.
                or use 'helm install --set version=2.0.0'   # override individual value
            Helm can also be used as a template engine, by creating template.yaml with {{ .Values.[name] }} and
            values.yaml  [name]: value. Can be used in CI/CD. Also helpful when deploy same set of application for
            different clusters(pod, dev, qa) Helm chart structure: mychart/ (top level name of chart), Chart.yaml
            (meta info about chart), values.yaml (values for the template files), chart/ (chart dependencies folder),
            templates/ (actual template files)
            Helm also provide release management. in Helm 2, there is client (helm cli) and server(Tiller). when client
            send request to Tiller, Tiller create components and runs inside k8s cluster. When create or change
            deployment, Tiller store copy of configuration client sent and keep track of all chart executions.
            When using 'helm upgrade [chartname]', changes are applied to existing deployment instead of create new one.
            If something went wrong, use 'helm rollback [chartname]' to change back to previous deployment. Tiller has
            too much permission, cause security issue. So in Helm 3, Tiller got removed.

        installation: minikube, kubeadm, yum, binary, 3rd party, cloud(Amazon EKS)
        minikube simulate kubernetes in virtual environment for learning
            https://minikube.sigs.k8s.io/docs/start
            minikube creates a virtual box that nodes run inside, it has one node with master and worker processes for
                testing purpose. need install a hypervisor(virtual box). minikube command only used for start/delete
                cluster
            kubectl: command line tool for kubernetes cluster to interact with Api server. other options are user
                interface and API
            minikube start     # start cluster
            create deployment.yaml   # config for pods version, kind(dev/prod environment), meta data, spec: selector
                (specify pod data transfer to), replica, template: meta data, container name, image, resources(cpu,
                memory), port, (type:NodePort  for nodeport service)

        deployment.yaml    contain 3 part: metadata, spec, and status(auto generated by k8s (check desire state=actual
            state, status get constantly updated. status retrieved from master etcd))
            kubernetes will also generate other code in yaml file once deployed, need to be cleaned if want reuse
            usually store deployment file in application code or separate configure git repository.
            inside spec template, define the pod configuration has its own metadata and spec
            use pod template metadata labels (any key-value pair) and spec selector matchLabels to connect pod and
                deployment.
            deployment metadata label are connect with service selector
            use yaml online validator to validate indentation

        deployment.yaml
        apiVersion: apps/v1
        kind: Deployment        # create deployment
        metadata:
            name: myapp
            labels:
                app: nginx
        spec:
            replicas: 3
            selector:
                matchLabels:
                    app: nginx   # connect with template metadata labels
            template:               # configure for pod
                metadata:
                    labels:
                        app: nginx
                spec:
                    containers:
                    - name: nginx               # can have multiple container in pod
                                                # - name: log-collector    with different containerPort
                        image: nginx:1.10   # specify docker image (can be on docker hub)
                        resources:      # optional
                            limits:
                            memory: "128Mi"
                            cpu: "500m"
                        ports:
                        - containerPort: 5000
                        env:
                        - name: USER
                            value: root
                        - name: PASSWORD           # name should be retrieved from docker hub image configuration if any
                            valueFrom:                          # retrieve secret value from secret service
                                secretKeyRef:
                                    name: nginx-secret          # match secret service name
                                    value: password-secret      # retrieve from secret service via key
                        - name: URL
                            valueFrom:                          # retrieve secret value from secret service
                                configMapKeyRef:
                                    name: nginx-configmap       # match configure map service name
                                    value: my_url               # retrieve from configure map via key
        ---                     # new document starting
        apiVersion: apps/v1
        kind: Service           # create service
        metadata:
            name: np-service    # NodePort to access server in cluster
        spec:
            selector:
                app: nginx      # connect with deployment metadata labels
            type: NodePort      # only for node port, need type for external service, internal service type is DEFAULT,
                                # no need specify
                # type: LoadBalancer  # for external service
            ports:
            - port: 80                          # service port
                targetPort:5000                 # pod port, match containerPort
                nodePort: 30080                 # optional. assign outer network port(for browser), range 30000-32767
                                                # automatically assigned if not specified
                                                # access via external service
                                                # or instead of nodePort and having type, use default type and assign
                                                # ingress to have https://myapp.com instead of ip:port

            ports
                - protocol: TCP
                    port: 80                    # service port
                    targetPort:5000             # pod port, match containerPort

            ports:
                - name: mongodb                 # if service has multiple port for multiple pod application, need have
                    protocol: TCP                   # - name
                    port: 27017
                    targetPort: 27017
                - name: mongodb-exporter
                    protocol: TCP
                    port: 9216
                    targetPort: 9216

        Secret: store secret (encrypt) values
        my-secret.yaml              # create secret
        apiVersion: apps/v1
        kind: Secret
        metadata:
            name: nginx-secret
        type: Opaque            # key-value pairs, there are other type of secret
        data:
            password-secret: VsaTAfC=          # get encoded data in terminal echo -n 'user_name_encoded' | base64
        ---
        apiVersion: v1
        kind: Secret
        metadata:
            name: myapp-secret-tls
            namespace: default         # same namespace as ingress
        data:
            tls.crt: base64 encoded cert
            tls.key: base64 encoded key

        kubectl apply -f my-secret.yaml     # need to run secret first before it can be referenced


        ConfigureMap: store global values
        my-map-config.yaml              # create configure map (store global variable)
        apiVersion: apps/v1
        kind: ConfigMap
        metadata:
            name: nginx-configmap
            namespace: my-namespace             # assign to specified namespace instead of default
        data:
            my_url: np-service                  # my_url: np-service.namespace_name  if located in namespace

            kubectl apply -f my-secret.yaml

        Ingress controller: use following configuration to forward traffic to internal service
        my-ingress.yaml              # create configure map (store global variable)
        apiVersion: apps/v1
        kind: Ingress
        metadata:
            name: myapp-ingress
        spec:
            tls:                            # required for HTTPS request
            - hosts:                        # add secret
                - myapp.com
                secretName: myapp-secret-tls
            rules:
            - host: myapp.com               # valid domain name and map to entrypoint of node
                http:                       # incoming request gets forward to internal service via http
                    paths:                  # part of url after ip:port/
                    - backend:              # single path
                        serviceName: np-service   # internal service name, has type ClusterIP with no external IP
                        servicePort: 80

                    - path: /analytics      # for multiple paths
                        backend:
                        serviceName: analytics-service
                        servicePort:3000
            or with subdomain
            -host: analytics.myapp.com
                http:
                    paths:
                    - backend:
                        serviceName: analytics-service
                        servicePort:3000

        use service(NodePort) to access server in cluster
        minikube start --vm-driver=hyperkit     # start cluster
        minikube status                         # get host,kubelet, apiserver, kubeconfig running status
        kubectl create -h                       # -h check document
        kubectl version                         # check client/server version

        kubectl create deployment [name] --image=image        # start deployment, provide deployment name and image name
                  #  kubectl create deployment nginx-depl --image=nginx           # [--dry-run][options]
                                                # if not specify deployment.yaml, will automatically generate one
                                                # can use for services and volume as well
        kubectl apply -f [file name]            # same as create but no need write name, image, options
                                                # run command again with file update to start new deployment
                                        # --namespace=my-namespace   to deploy in specified namespace instead of default
        kubectl edit deployment [name]          # get deployment.yaml file, after save the change of deployment file,
                                                # old pod will terminate and new pod will start automatically
        kubectl delete deployment [name]        # delete deployment
        kubectl delete -f [file name]           # delete deployment

        kubectl logs [pod name]                 # log to console for pod
        kubectl describe pod [pod name]         # get addition information of pod: type, reason, age, from,
                                                    # message (state change)
        kubectl describe service [service name]     # show target port and endpoints(pod ip:port)
        kubectl exec -it [pod name] --bin/bash  # get interactive terminal (get inside pod container)   exit  to quit

        kubectl get pod                         # check pod name, ready, status, restarts, age
                                                # pod name is deployment name-replicaset hash-some hash
                                                # -o wide  to get more information: ip address
        kubectl get nodes                       # check nodes name, status, roles, age, version
        kubectl get services                    # check service name, type, cluster-ip, external-ip, ports, age
        kubectl get deployment                  # check deployment name, ready, up-ro-date, available, age
                                                # -o yaml    get deployment yaml file with status
                                                # -n my-namespace   for access item not in the default namespace
        kubectl get replicaset                  # check replica name, desired, current, ready, age
        kubectl get configmap                   # check configure map: name, data, age
        kubectl get all                         # check pods, deployment and replica set

        kubectl apply -f deployment.yaml        # start/update (blue/green) deployment
        kubectl get service                     # check deployed service
        minikube service service-name           # assign a public ip address to the external service and open service
                                                # in browser, port will be nodePort: 30080, defined in deployment
        kubectl delete -f deployment.yaml       # delete deployment

        kubectl cluster info                    # get kubernetes public configuration map information
        kubectl get namespace                   # print existing namespace
        kubectl create namespace [namespace name]   # create custom namespace
            # or create with configuration file
                apiVersion: v1
                kind: ConfigMap
                metadata:
                    name: nginx-configmap
                    namespace: my-namespace
            data:
                url: np-service


        minikube addons enable ingress           # start k8s nginx implementation of ingress controller for minikube.
            # kubectl get pod -n kubesystem      # check nginx-ingress-controller
            after create ingress file
            apply -f my-ingress.yaml        kubectl get ingress   # get assigned Address of ingress
            inside /etc/host    add:   ingress IP address  myapp.com
            then in browser able to visit http://myapp.com
            kubectl describe ingress my-ingress   # show default backend to handle no rules match and return 404 page
                can create a service with name default-http-backend and same port to override default action


        template yaml config
        apiVersion: v1                                              values.yaml
        kind: Pod                                                   name: my-app
        metadata:                                                   container:
            name: {{ .Values.name }}                                    name: my-app-container
        spec:                                                           image: my-app-image
            containers:                                                 port: 9001
            - name: {{ .Values.container.name }}
                image: : {{ .Values.container.image }}
                port: {{ .Values.container.port }}


        persistence volume yaml config
        apiVersion: v1       # nfs here,        can use local hard disk and cloud storage as well
        kind: PersistentVolume
        metadata:                                   # local storage         # Google cloud
            name: pv-name                                                   # labels:
                                                                            #   failure-domain.beta.kubernetes.io/zone:
                                                                            #   us-central1-a__us-central1-b
        spec:
            capacity:
                storage: 6Gi
            volumeMode: Filesystem                                         # no need volumeMode
            accessModes:
                - ReadWriteOnce
            persistentVolumeReclaimPolicy: Recycle    # Delete              # gcePersistentDisk
            storageClassName: slow                    # local-storage       #    pdName: my-data-disk
            mountOptions:                             # local:              #    fsType: ext4
                - hard                                #   path: /mnt/disks/ssd1
                - nfsvers=4.0                         # nodeAffinity:
            nfs:                                      #   required:
                path: /dir/path/on/nfs/server         #      nodeSelectorTerms:
                server: nfs-server-ip-address         #      - matchExpressions:
                                                      #         - key: kubernetes.io/hostname
                                                      #            operator: In
                                                      #            values:
                                                      #            - example-node


        Persistent Volume Claim     # claim persistent volume
        kind: PersistentVolumeClaim
        apiVersion: v1
        metadata:
            name: pvc-name
        spec:
            storageClassName: manual
            volumeMode: Filesystem
            accessModes:
                - ReadWriteOnce
            resources:
                requests:
                    storage: 10Gi
            storageClassName: storage-class-name    # same as storage class config

        apiVersion: v1      # pod configure
        kind: Pod
        metadata:
            name: mypod
        spec:
            containers:
                -name: myfrontend
                image:nginx
                volumeMounts:       # add this for Persistent Volume Claim
                - mountPath: "/var/www/html"        # application can access the mounted data here: "/var/www/html"
                    name: mypd
            volumes:                # add this for Persistent Volume Claim
                -name: mypd
                    PersistentVolumeClaim:
                        claimName: pvc-name     # match the claim name

            volumes:                # mount secret or configMap
                -name: config-dir
                    ConfigMap:
                        claimName: bb-configmap


    Storage Class
    apiVersion: storage.k8s.io/v1
    kind: StorageClass
    metadata:
        name: storage-class-name
    provisioner: kubernetes.io/aws-ebs      # each storage has own provisioner
    parameters:
        type: io1
        iopsPerGB: "10"
        fsType: ext4


    kubeadm
            disable swap, cpu 2+ core, 2G+ memory, centos 7.x, machine in cluster able to communicate
            systemctl stop/disable firewalld
            setenforce 0  # close selinux temporary (turn off linux security)
                # or sed -i 's/enforcing/disabled/' /etc/selinux/config   # permanent, need restart machine
            swapoff -a    # temporary disable swap    or sed -ri 's/.*swap.*/#&/' /etc/fstab   # permanent, need restart

            cat >> /etc/hosts << EOF                    # add hosts in master
            192.168.172.131 k8smaster
            192.168.172.132 k8snode
            EOF

            cat > /etc/sysctl.d/k8s.conf << EOF          # set net bridge parameter
            net.bridge.bridge-nf-call-ip6tables = 1
            net.bridge.bridge-nf-call-iptables = 1
            EOF
            sysctl --system  # activate setting

            yum install ntpdate -y                      # sync nodes and master time
            ntpdate.time.windows.com

            kubeadm init
            kubeadm join master ip:port


'''