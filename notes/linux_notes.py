'''
    Redhat (good and expensive)  CentOS (similar to redhat)   Ubuntu (good for pc)
    Termius  ssh tool   visit public ip address

    $ who [ -a |  -b -d -i -l -m -p -q -r -s -t -u  -w -A -H -T -X ] [ File  ]  # all login at this time
    $ who am { i | I }     # only current login
    o: Name [State] Line Time [Activity] [Pid] [Exit] (Hostname)
    The who command displays information about all users currently on the local system.
    $ w                      # show detailed version with cpu usage, idle time...
    $ last                   # show recent login record
    $ exit / logout          # exit connection
    $ adduser                # create user
    $ passwd                 # change password
    $ userdel                # delete user
    $ su username            # switch user to username
    $ date   $ cal           # check date and calender
    $ write user [ttyname]   # send message to a user
    $ wall [ message ]       # send message to all (mesg set to yes)
    $ mesg [ny]              # whether receive message
    $ clear                  # clear console output
    $ man / info / --help    # show help documentation
    $ history                # show command history   -c clear history
    $ reboot / init 6        # restart system
    $ shutdown / init 0      # shutdown system      shutdown now: immediately shutdown


    most used command
    $ pwd                    # print working directory
    $ cd                     # change directory
    $ ls                     # show current directory items, -a show hidden(.xxx.py), -l more info  -R recursive
    $ cat                    # concatenate multiple file / check file content    -n  add line number
                             cat -n taobao.html | more     |  put leftside output as input and pass to right side
    $ touch                  # create empty file or change last visited date
    $ mkdir                  # create directory    -p  recursive create
    $ rm                     # remove    -f  force    -r  recursive
    $ rmdir                  # delete empty folder
    $ wget                   # retrieve file from web   -O  rename file
    $ gzip / gunzip          # zip and unzip file (.gz/.tgz)
    $ xz                     # zip and unzip file (.xz)    -z  zip    -d  unzip
    $ tar                    # (un)archive file   -xvf  unarchive     -cvf  archive (combine files into one)
    $ wc                     # count   -l  count lines     -w  count words       -c  count characters
    $ sort                   # sort file   -r  reverse order
    $ uniq                   # remove duplicate content in the file
    $ head / tail            # show top and end lines of content
    $ more / less            # show file in pages
    $ diff                   # show difference between files
    $ chmod xxx  file.txt    # 4 read, 2 write, 1 execute, add together value for owner, same group, other group user
                                    # 777 all access
    $ cp                     # copy file/directory   cp a.txt folder/  copy a.txt to folder directory
                             # cp a.txt folder/b.txt   copy and rename      cp -r folder /   copy directory
    $ mv                     # move file/directory    rename file if under same directory
    $ ln                     # create hard shortcut  ln a.py /root/b: create shortcut of a.py under root with name b
                             # create reference of file (no extra space) but will still live if one reference removed
                             # create soft shortcut  ln -s a.py /root/b, remove original file will delete shortcut
    $ alias ll='ls -ls       # create alias ll
    $ unalias ll             # delete alias ll
    $ find / -name "*.html"  # find all .html file under / (path)    add -delete  to delete found files
                             # -size +10M    find all  file under / greater than 10M size
                             # -perm 664 search file under / with read,write for owner,group, and read for other
                             # -type file type
                             # -atime  read date   -mtime   modify date    -ctime 2  create date with in 3 days
    $ ssh                    # ssh username@ip     remote instance username and ip address
    $ scp                    # scp home/a.py root@12.123.456.78:/home/dir   # secure copy file
                             # scp code/ root@12.123.456.78:/root/   # copy directory
    $ sftp                   # sftp root@12.123.456.78   # connect to remote machine
                             # lls   local  ls     lcd ..   local change to previous directory
                             # put code.py  # upload file to other machine    -r for directory
                             # get code.py   # get file from other machine
                             # help show manual
                             # bye / quit  exit sftp (secured file transfer protocol)
    $ ping                   # ping www.baidu.com    check ip availability
                             # PING TO DEATH   DDoS(distributed deny of service)  TCP Flood
    $ ifconfig /ip addr      # show network card eth0 info (inet private ip ) lo localhost
                             # 127.0.0.1 localhost  (set under /etc host file)
    $ netstat                # network status   -n number format    -a  all   -p process
                             # stream: tcp type    ugram: udp type    ip type
    $ kill                   # kill 1211   kill  process 1211  -9 force
    ps /jobs                 # check running process   -ef all process
    fg                       # move command to foreground    fg %1    jobs get id [1]+ Running
    Ctrl+z                   # stop foreground job and move to background
    bg                       # start the command in background    bg %1
    top                      # process manage     q to quit

    |                        # put left side command output as input and pass to right side
    grep                     # search string  support regular expression
                             # ls -R | grep example   search file name under directory
                             # grep -E "\<\/?script.*\>" index.html  search <script> tags in index.html
                             #   grep -E "\<\/?script.*?\> index.html    non greedy search
                             # grep redis | grep -v auto  # search result have redis but no auto
                             cut
                             f4
    >  /  >>                 # redirect output to file    >> append
    2>  /  2>>               # error redirect output to file    >> append

    Ctrl+d                   # end input
    Ctrl+c                   # stop running process
    Ctrl+w                   # delete part of command
    Ctrl+a                   # move cursor to beginning
    Ctrl+e                   # move cursor to end

    tab   # command and path auto fill


    Linux install package
    1. use package managing tool: yum / rpm / apt     for linux/ redhat / unbuntu
    2. source code build install:  gcc (c compiler) / make

    yum search nginx          # search package name
    yum install nginx         # install package (no need specify 32/64 bit )  -y  yes for questions
    yum info nginx            # show installed package info
    yum update nginx          # update package
    yum update                # update all packages
    yum remove nginx          # uninstall package
    yum list installed        # show all installed package
    yum list installed | grep nginx       # show installed package contain nginx

    install python via source code
    1. wget https://www.python.org/ftp/python/3.7.3/Python-3.7.3.tar.xz
    2. xz -d Python-3.7.3.tar.xz
    3. tar -xvf Python-3.7.3.tar
    4. yum -y install zlib-devel bzip2-devel openssl-devel ncurses-devel sqlite-devel tk-devel db4-devel
        libcap-devel xz-devel libffi-devel  libcurl-devel
    # Devel libraries typically contain development header and debug resources that are not necessary
        for the end-user runtime.
    5. cd Python-3.7.3   ./configure --prefix=/usr/local/python37 --enable-optimizations
    6. cd ~      vim .bash_profile   add   PATH=...:/usr/local/python37/bin        source .bash_profile
        python3 --version
    7. cd ~    touch .vimrc     vim .vimrc   # vim file setting  (optional)
        set nu    syntax on   set ts=4    set expandtab    set autoindent    set ruler   set nohls




    vim index.html

    i                         # insert mode
    esc                       # back from insert mode to command mode
    :  /                      # change to last line mode
    enter                     # back from last line mode to command mode
    :q  :q!   :wq             # quit   force quit    save quit
    shift + zz                # save quit
    dd  100dd                 # delete one line   (100 lines from cursor line)
    u / Ctrl+r                # undo / redo
    yy    10y                 # copy one line  10 lines
    p     10p                 # paste    10 lines

    G    300G                 # cursor move to the end of file    end of line 300
    GG                        # cursor move to the front
    h j k l   30h             # move cursor to left down right up     left 30
    Ctrl+y / Ctrl+e           # move cursor up down one line
    Ctrl+f / Ctrl+b           # move cursor up down one page
    0 / $ / w                 # move cursor to front/end of line/ next word

    map <F2> gg9999dd         # map key to function in command mode (create shortcut)
    ignoremap _main if __name__ == '__main__':    # custom shortcut in coding

    $ q                      # create macro (record repeatable action)
                             # qa create macro named a, i insert mode and make change, esc, q stop recording
                                # @4a  (a is the macro name)  repeat 4 times action at cursor
    $ /0   ?0                # search from top / bottom string 0    n for next   N for previous
    $ 1,100s/0/1000/cg       # replace 0 with 1000 in line 1 to 100  press Y to confirm
                             # c : confirm   g: global (find all in one line)  i: ignore case  e: ignore error


    vim a.py b.py # open multiple files    :ls  #show all files opened    :b 2  # switch to 2nd file
    :vs / :sp                 # vertical / horizontal split screen
    Ctrl+w twice              # switch window
    :qa                       # exit all windows
    vim -d a.py b.py # open multiple files and highlight difference

    accident quit
    R                         # recover, then  rm .xxx.py.swp  to delete recovery file

    :set nu / :set nonu       # add line number   remove line number
    :set ts=4                 # set tab = 4 spaces
    :syntax on / :syntax off  # highlight grammar keyword    close highlight
    :set ruler / :set noruler # show cursor location   close ruler
    set autoindent            # auto indent when previous line have indent

    gcc --version


    Nginx                     # web server (http server)
                              # need create security group: custom tcp  80/80  0.0.0.0/0  allow all ip visit
                              # nginx            command to start server
                              # or service nginx start # ubuntu and centos 6
                              # systemctl start/stop/restart/status/enable/disable nginx
                                # enable  start nginx automatically when start machine
                              # nginx -s stop    stop server

    systemctl start firewalld    # use own firewall
    firewall-cmd --add-port=80/tcp --permanent
    systemctl restart/stop firewalld

    Github      https://gitee.com    https://coding.net
    https://git-scm.com    tarballs
    wget https://mirrors.edge.kernel.org/pub/software/scm/git/git-2.21.0.tar.xz
    xz -d git-2.21.0.tar.xz   tar -xvf git-2.21.0.tar    cd git-2.21.0
    ./configure --prefix=/usr/local   # set install path
    make && make install

    git config --global user.email "you@example.com"
    git config --global user.name "your name"
    git init  # initialize git repository, add .git
    git add file / git add .   # add file cache
    git status                 # check status
    git commit -m "mesg"       # commit code move cache to repository
    git push                   # sync with server
    git pull                   # receive code changes in server
    git checkout -- a.py / .   # revert deleted (all) file
    git log                    # show previous commit history
    git reset --hard <ID>  HEAD^     # revert to previous commit specific commit with id
    git reflog                 # show forward commit

    git clone url              # clone project to local
'''
