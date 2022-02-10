'''
    Airflow: open source platform to build and run workflows(a sequence of tasks, started on a scheduler or triggered by
        an event, for ETL, big data processing pipeline, machine learning, etc.) enabling scheduling (based on metadata
        (in metadata base) and DAG) and monitoring. User can define a dependent task set (DAG, directed acyclic graph),
        execute order depend on dependencies. Airflow also provide command line tool and web UI for easy scheduling,
        able to monitor real time running status. Airflow has scheduler, webserver(Gunicorn framework), metadata
        database (default SQLite, support mysql, postgres, save DAG, task definition, run history, user, privilege),
        worker (process that handle task from executor)

    Dynamic: Airflow pipeline use python for configuration, allow dynamic instantiated generate pipeline
    extendable: can define own operators, executors and extension library
    elegant: Airflow pipeline is very clear and concise, use Jinja template engine, put script parameter into core of
        Airflow
    Elastic: Airflow use modularization framework, use messaging queue to coordinate large amount of workers

    Scheduler: monitor all tasks and DAGs, trigger downstream task instances once their dependencies are complete. Have
        a subprocess monitor and sync with DAGs. Default every minute it checks whether any tasks can be triggered.
        steps: check for DAG need a new DagRun(instantiation of DAG) and create; check batch of DagRun for any task
        instances able to schedule, otherwise complete DagRuns; select task instance and enqueue for execution if under
        pool limit. Inside Scheduler has an executor to trigger worker. DAG files are under DAG directory. Scheduler
        will read in DAG file, process the DAG and write it to Metadata database for faster retrieval by webserver and
        scheduler DAG execution queue.
        Executor support many implementationL Sequential, Local (multiprocess), Celery, and Dask(distributed numpy),
        Kubernetes Executor
        DagFileProcessor: parses DAGs into serialized_dags table
        In cluster settings, Airflow has webserver and (multiple) scheduler to access DAG files and use Celery worker(on
        different server) to execute task based on queue broker (rabbitMQ/ Redis) assigning task.
        scheduler: do_scheduling(), processor_agent.heartbeat(), heartbeat(), timed_events.run()
        DagFileProcessor: collect_result_from_processor, start_new_processes, send heartbeat, refresh_dag_dir
        multiple scheduler(active, active) use db task lock synchronous to ensure only one scheduler access a task at a
            time, and only one scheduler can access resource like pool at a time for update without wait

    Install: only for linux and max. 1, create virtualenv   python3 -m venv /path/to/new/virtual/environment
        2. install airflow locally with script at https://airflow.apache.org/docs/apache-airflow/stable/start/local.html
        3. start and stop Airflow webserver/scheduler

    airflow db init     # initialize the database
    airflow webserver --port 8080   # start the web server,  add '-D' to run it as a daemon
    airflow scheduler     # start scheduler   '-D' as daemon
        echo "export AIRFLOW_HOME=/Users/cai/workspace/cai.com/airflow" >> ~/.bashrc    # add environmental variable
        echo "export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES" >> ~/.bashrc    # additional for mac
    ps aux | grep airflow-webserver     kill pid  (master [airflow-webserver])

    User Interface
        DAGs:  DAG(name), owner, runs(success, fail), schedule(chron), last run(time), recent tasks, Actions(run,
            refresh, delete), links(fast link same click inside DAG name: tree view, graph view, calendar view, task
            duration, task tries(# runs till success), landing times, gantt(time vs task graph with rectangle of tasks
            represent time start/end), details, code), update(mark success, failure, clear) task
        Tasks: click on the square box in side DAG view, can see instance details, log,

    Create DAG
        add dags folder under airflow installed path, in the airflow.cfg specify the dags_folder
        create dag python file inside dags folder, copy the code in UI
        update with DAG()  section for id, args, description, schedule_interval, start_date, tags; update default_args
            # change environment and run dag file directly, file correct if no error
            from airflow.operators.python import PythonOperator
            with DAG(dag_id='my_task_flow', default_args=default_args, description='xx', schedule_interval='5 5 * * *',
                start_date=days_ago(2), tags=['harry.com'], catchup=False, max_active_runs=1) as dag:
                    # catchup=False, max_active_runs=1  solve dead lock, specify it to synchronous work_flow
                        # catchup whether need to run multiple times for previous missing runs or one time is enough
                task_a = PythonOperator(task_id = 'task_name', python_callable = my_func)   # func1 is a function
                task_a >> task_b   # to create sequential dag steps
                t0 >> t1 >> [t2, t3]  # list of dependencies
                t0.set_downstream(t1)  # set_upstream(t1)

        UI will display changes inside DAG folder for any updates with auto_refresh enable, click run in UI to run DAG
            or click off to on to turn on scheduler (ex. run once a day after the full time(24hr) period is covered)

    Create variable
        # inside UI under Admin tab -> Variables add record  add key, val, description  # key: var_a   val: my_val
            #special key like API_key and password val will display as ******
        var = Variable.get('var_a')
        var_json = Variable.get('var_b', deserialize_json=True)

        Also when run in UI there is a trigger DAG w/ config option to trigger job with input param
        during trigger add configuration JSON {'var_c': ['a',2]}    during define function add input param **context
        def my_func(**context):
            var_param = context["dag_run"].conf.get('var_c')
        task_b = PythonOperator(task_id = 'task_name', python_callable = my_func, provide_context=True)  # add context



    Change Sqlite to MySQL for metadata database
        pip install mysql-connector-python
        msql: create database airflow character set utf8;
        in airflow.cfg change sql_alchemy_conn = mysql+mysqlconnector://root:123456@127.0.0.1:3306/airflow
        airflow db init
        airflow users create --username airflow --firstname Harry --lastname Potter --role Admin  --email har@gmail.com
            # create airflow user
        kill pid xxx   # kill webserver and scheduler master   ps aux | grep airflow
        delete airflow-webserver.pid and airflow-scheduler.pid  in the project folder
        airflow webserver --port 8080 -D
        airflow scheduler -D

        inside UI admin tab connections, edit connection info(ConnId, Conn type, host, port, schema, login, password)
            host is host.docker.internal if use docker
        from airflow.hooks.base_hook import BaseHook
        from airflow.providers.mysql.operators.mysql import MySqlOperator
        conn = BaseHook.get_connection('conn_id')   # conn_id is ConnId defined in UI, use hook to grab connection info,
                                                    # so no need to type out password in code
        db = mysql.connector.connect(host=conn.host, host=conn.host, user=conn.login, password=conn.password, database=
            conn.schema, port=conn.port)

        task_c = MySqlOperator(task_id = 'task_name', mysql_conn_id='conn_id', sql='steps.sql',dag=dag)
            # add task for MySQL using sql file

    XComs: allow tasks to talk to each other(push and pull  some value in xcom from different tasks)
        Many operators will auto-push their result in XCom key called return_value if any return value
        def my_func2(**context):
            var_param = context["dag_run"].conf.get('var_c')
            return var_param  # many operator push automatically with return statement
                # val = context['ti'].xcom_push(key='my_key', value='my_value')   # context['ti']: task instance
                (task_ids='pushing task')   # context['ti']: task instance
        t2 = PythonOperator(task_id='task_xcom_push', python_callable=my_func2, provide_context=True, dag=dag)
        then in another def my_func3(**context):
            my_v = context['ti'].xcom_pull(task_ids='task_xcom_push', key='my_key')
                # no need key if from return statement

    Robustness (retry, email, alert)
        email:
            in airflow.cfg  [smtp] section
                smtp_host=smtp.gmail.com
                smtp_starttls = False
                smtp_ssl = True
                smtp_user = harry@gmail.com
                smtp_password =
                smtp_port = 456   #ssl 456, tls 587
                smtp_mail_from = airflow@gmail.com
                smtp_timeout = 40
                smtp_retry_limit = 5
            from airflow.operators.email_operator import EmailOperator
            email_task = EmailOperator(task_id='send_email', to='harry@gmail.com', subject='{{ ds }}', html_content=
                """<h3>Email Test </h3>""", dag=dag)    # {{ ds }} macro for date, only some attribute support macro,
                                            # need to check document whether attribute is templated to put macro inside
                                            # {{ }} macro can return a string or object (need to specify object.attr in
                                            # order to print something
        for alert when failure, change the default_args in airflow.cfg
            default_args= {'email':[harry@gmail.com], 'email_on_failure':True, 'email_on_retry':True, 'retries':1,
                'retry_delay':timedelta(seconds=30)}        # retry is task retry not email


    Provider
        Airflow provide basic framework: scheduler, worker, executor, user interface. How to deal with other service(
        database, cloud, application API) are implemented by open community or provider.
        reference to online airflow document for steps, need install provider package
        create connection with connection id in user interface, install corresponding package
        s3 = S3ListOperator(task_id='s3_list', bucket='xxx', prefix='xx/xxx', delimiter-'/', aws_conn_id='conn_id')

    Sensor
        monitor the status change of upstream change, start some task after the status changed. sensor is depend on
        different provider (inside their package)
        s3_s = S3KeySensor(task_id='s3_sen', bucket_key='xxx/{{ ds_nodash }}/*.csv', wildcard_match=True, bucket_name=
            'xxx', aws_conn_id='conn_id', timeout=18*60, poke_interval=30, dag=dag)

    DAG dependencies
        dependencies between tasks in a DAG can be defined through upstream and downstream. Dependencies between DAGs
        are defined with either trigger (TriggerDagRunOperator) or waiting (ExternalTaskSensor)
        trigger_next = TriggerDagRunOperator(trigger_dag_id='abc', task_id='t0', execution_date="{{ds}}", wait_for_
            completion=False)   # at dag pre_dag, task task_id 't0', trigger next dag with id 'abc'
        pre_dag >> trigger_next

    Dynamic DAGs
        # python no need to declare a new object( a = new Object() )
        t0, t1 = PythonOperator(...), PythonOperator(...)
        options = ['branch_a', 'branch_b']
        for option in options:
            t = PythonOperator(task_id=option)
            t0 >> t >> t1

    Logging
        need put logging into remote directory (local logging will be gone if using container, or not sure which worker
            on which server).
        in airflow.cfg  [logging] session:
            remote_logging=True
            remote_log_conn_id = aws_s3   # connection id created in user interface of airflow
            remote_base_log_folder=s3://harry/airflow/logs/

    Airflow CLI and API
        airflow -h              # help document for available command
        airflow info            # show available provider, path info, system info
        airflow cheat-sheet     # common command and description

            airflow dags list
            airflow dags unpause dag_id
            airflow dags trigger dag_id
            airflow dags list-runs
            airflow  tasks run dag_id task_id "2021-09-01 11:55:00"  # execution date

        airflow.cfg: change from deny_all to          auth_backend = airflow.api.auth.backend.basic_auth
            # core section is more important
            # allow web api

        metadata database
            tables: user, role, permission, dag, dag_code, connection, dag_run, job, log, serialized_dag, sla_miss,
                sensor_instance, task_instance, xcom
            sometime modify in metatable easier than UI



'''