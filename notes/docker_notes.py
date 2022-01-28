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
                ENTRYPOINT ["nginx"]    ENTRYPOINT [ "curl", "-s", "http://ip.cn" ]
            CMD python3               # command to execute to run other program, when docker container is running
                                      # param can be override by docker run
                CMD <shell command>
                CMD ["<executable file or command >","<param1>","<param2>",...]   # recommended way
                CMD ["<param1>","<param2>",...]  # provide ENTRYPOINT program default param
                    # only last CMD will be executed, previous one inside same docker file will be ignored
                    CMD /bin/sh -c 'nginx -g "daemon off;"'    CMD ["-g","daemon off;"]   CMD ["/usr/bin/wc","--help"]
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
            docker-compose down  # stop and remove container    --volumes   delete volumes

            docker-compose.yml
            version: '2'
            command: [bundle, exec, thin, -p, 3000]  # optional, overwrite default command after container initialized
                # command: bundle exec thin -p 3000
            services:
                web:   # custom service name
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
        Service: each pod's service get a fix IP, even the container died won't change the service IP associate with
            this pod, will reassign a replica with same service IP instead of assigning a new IP each time container
            failed. And pods communicate to each other via service. All the replicas of application share a service
            with same service IP. Service also act as a load balancer.
        Ingress: The request first goes to ingress, so user can access page via https://my-app.com, then ingress route
            traffic to service IP address (http://ip:port).
        Configure Map: external configuration of application, connect to the pods. so if need to change the config of
            the service, no need to rebuild the image with new application properties file.
        Secret: similar to configure map, (credential, certificate) but not stored in plain text format. secret are
            accessible by pods
        Volume: attach a (local/remote) physical storage to pod. So all the data will be persisted with pod restarts.
            Kubernetes don't manage data persistence, so you have to backup with other solution.
        Deployment: blue print for pods, another layer of abstraction over pods. User specify number of replica in
            deployment. Deployment can only replicate application, not database(has state(changed data))
            deployment manage all the replica set (replica of pod). user only need manage deployment, any thing below is
            handled by kubernetes
        Stateful set: used to replicate database for pods. make sure synchronized data across replica. stateful set is
            harder than deployments, so it's common to host database outside kubernetes cluster.
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
            labels:             # create alias
                web: nginx
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
                    - name: nginx
                        image: nginx:1.10   # specify docker image (can be on docker hub)
                        resources:      # optional
                            limits:
                            memory: "128Mi"
                            cpu: "500m"
                        ports:
                        - containerPort: 5000
        ---
        apiVersion: apps/v1
        kind: Service           # create service
        metadata:
            name: np-service   # NodePort to access server in cluster
        spec:
            selector:
                app: nginx      # connect with deployment metadata labels
            type: NodePort      # only for nodeport
            ports:
            - port: 80                          # service port
                targetPort:5000                 # pod port, match containerPort
                nodePort: 30080    # optional assign outer network port, automatically assigned if not specified

            ports
                - protocol: TCP
                    port: 80                    # service port
                    targetPort:5000             # pod port, match containerPort

        use service(NodePort) to access server in cluster
        minikube start --vm-driver=hyperkit     # start cluster
        minikube status                         # get host,kubelet, apiserver, kubeconfig running status
        kubectl create -h                       # -h check document
        kubectl version                         # check client/server version

        kubectl create deployment [name] --image=image        # start deployment, provide deployment name and image name
                  #  kubectl create deployment nginx-depl --image=nginx           # [--dry-run][options]
                                                # if not specify deployment.yaml, will automatically generate one
                                                # can use for services and volume as well
        kubectrl apply -f [file name]           # same as create but no need write name, image, options
                                                # run command again with file update to start new deployment
        kubectl edit deployment [name]          # get deployment.yaml file, after save the change of deployment file,
                                                # old pod will terminate and new pod will start automatically
        kubectl delete deployment [name]        # delete deployment
        kubectl delete -f [file name]           # delete deployment

        kubectl logs [pod name]                 # log to console for pod
        kubectl describe pod [pod name]         # get addition information of pod: type, reason, age, from,
                                                    # message (state change)
        kubectl describe service [service name]     # show target port and endpoints(pod ip:port)
        kubectl exec -it [pod name] --bin/bash  # get interactive terminal (get inside pod container)   exit  to quit

        kubectrl get pod                        # check pod name, ready, status, restarts, age
                                                # pod name is deployment name-replicaset hash-some hash
                                                # -o wide  to get more information: ip address
        kubectrl get nodes                      # check nodes name, status, roles, age, version
        kubectrl get services                   # check service name, type, cluster-ip, external-ip, ports, age
        kubectrl get deployment                 # check deployment name, ready, up-ro-date, available, age
                                                # -o yaml    get deployment yaml file with status
        kubectrl get replicaset                 # check replica name, desired, current, ready, age


        kubectrl apply -f deployment.yaml       # start/update (blue/green) deployment
        kubectl get service                     # check deployed service
        minikube service service-name           # open service in browser
        kubectrl delete -f deployment.yaml       # delete deployment


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