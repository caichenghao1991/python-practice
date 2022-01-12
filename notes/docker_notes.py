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
        docker build -t redis1:v2 .   # build image . is context path if needed from pc running the container


        Docker Compose
            docker compose is a tool used for define and run application on multiple docker container. use YML file to
                configure all the application's service (version, services(ports, build, volume, links, args, target,
                dockerfile, context, command, container_name, depends_on, cap_add, cap_drop, cgroup_parent,
                deploy(endpoint_mode, labels, mode, replicas, resources, restart_policy, rollback_config, update_config)
                , devices, dns, dns_search, entrypoint, env_file, environment, expose, extra_hosts, healthcheck, image,
                logging, network_mode, restart, secrets, security_opt, stop_grace_period, stop_signal, sysctls, tmpfs,
                ulimits, volumes). use docker-compose up to start the application

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
'''