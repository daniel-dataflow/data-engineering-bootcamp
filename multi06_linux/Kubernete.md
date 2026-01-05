# Kubernetes



```bash
# 명령어, 리소스 이름 등 자동완성
source <(kubectl completion bash)

# terminal 실행 될 때마다 자동완성 코드 입력
sudo vim ~/.bashrc

source <(kubectl completion bash)
# :Wq!

source ~/.bashrc
```



#### 1. namespace

\- cluster 하나를 여러 개의 논리적인 단위로 나눠서 사용

```bash
kubectl get namespaces
kubectl get ns

# get namespaces 결과
NAME              STATUS   AGE
default           Active   30m
kube-node-lease   Active   30m
kube-public       Active   30m
kube-system       Active   30m

# namespace 지정하지 않으면 default
# default namespace에 아직 아무것도 없음
kubectl get pods
# kube-system에는 운영에 필요한 pod들이 저장
kubectl get pods -n kube-system

# namespace 생성
kubectl create ns test-system
kubectl get ns
# namespace 삭제
kubectl delete ns test-system
kubectl get ns
```



#### 2. pod

\- container 실행 및 배포 단위 

\- pod 안에 있는 container는 ip를 공유 (일반적으로 pod 하나에 container 1~2개)

```bash
# pod 생성
# 1. 명령어 사용
kubectl run pod01 --image nginx
# 생성된 pod 확인
kubectl get pods
kubectl get pods -o wide
# watch -n 1 kubectl get pods -o wide

# wide에서 보여진 ip
curl -sf 10.244.235.129
# ip addr -> tunl0 : calico에서 사용하는 터널 인터페이스
```

*vim pod.yaml*

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: pod02
spec:
  containers:
  - name: nginx-container
    image: nginx
    ports:
    - containerPort: 80
```

```bash
# pod 생성
# 2. yaml 사용
# pod의 name은 namespace에서 유일해야 함
# 명령어
# run : 없으면 만들어라.
# apply : 없으면 만들고, 있으면 없데이트해라.
kubectl apply -f pod.yaml

kbectl get pods
kbectl get pods -o wide

# pod 삭제
kubectl delete pod pod01
kubectl delete pod pod02

kubectl get pods
```



**static pod**

```bash
ls /etc/kubernetes/manifests/
# master node에 반드시 있어야 할 4개에 대한 yaml 파일들
# etcd.yaml  kube-apiserver.yaml  kube-controller-manager.yaml  kube-scheduler.yaml
# kubelet이 실행시키는 위의 4개 컴포넌트를 static pod라고 부름
# kubelet이 실행시키는 pod들을 static pod라고 부름!
kubectl get pod -n kube-system
# etcd-master
# kube-apiserver-master
# kube-controller-manager-master
# kube-scheduler-master
# 이 4개가 static pod
```



#### 3. lifecycle

**container lifecycle**

\- wating
	ContainerCreating : 이미지 다운로드 중 or 컨테이너 생성중
	CrashLoopBackOff : 컨테이너 재시작 대기
	ImagePullBackOff : 이미지 다운로드 재시도 중

\- Running : 실행중

\- Terminated
	Completed : exit code 0
	Error

*vim hello.yaml*

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: hello-pod
spec:
  containers:
  - name: hello-container
    image: busybox
    command: ["/bin/sh", "-c"]
    args:
    - 'for i in $(seq 1 5); do echo "Hello, World!"; sleep 5; done'
```

```bash
# 확인
# watch -n 1 kubectl get pods -o wide
kubectl get pods -w -o wide
# 새 창 열어서 실행
kubectl apply -f hello.yaml
# 로그 확인
watch -n 1 kubectl logs hello-pod
```

*busybox : 명령어 모음 이미지*



**pod lifecycle**

\- pending : 생성중
\- running : 실행중
\- succeded : 정상 실행 종료 (completed )
\- failed : 비정상 종료
\- unknown : pod가 올라간 worker node와의 통신이 불가하여 pod가 살아있는지 죽어있는지 알 수 없음

*lifecycle-pod.yaml*

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: lifecycle-pod
spec:
  containers:
  - name: lifecycle-container
    image: busybox
    command: ["/bin/sh", "-c", "exit 1"]
```

```bash
# pod 상태를 실시간으로 출력
kubectl get pods -w

# 새로운 창에서 pod 생성
kubectl apply -f lifecycle-pod.yaml
# pending -> cotainer creating 이후에 error -> crashroopbackoff 반복

# 상세 정보 확인
kubectl describe pod lifecycle-pod
```





**kubelet probe**

\- iveness probe : 실행되어있는지 확인, 진단 실패하면 재시작
\- readiness probe : "ready" 상태 확인 (요청에 맞게 처리해서 응답하는지), 진단 실패하면 서비스 요청을 보내지 않음
\- startup probe : 시작되는지 확인, 진단 실패하면 재시작

*liveness-pod.yaml*

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: liveness-pod
spec:
  containers:
  - name: liveness-container
    image: busybox
    command: ["/bin/sh", "-c"]
    args:
    - touch /tmp/healthy; sleep 30; rm -f /tmp/healthy; sleep 60
    livenessProbe:
      exec:
        command:
        - cat
        - /tmp/healthy
      initialDelaySeconds: 5
      periodSeconds: 5
```

```bash
kubectl apply -f exec-liveness.yaml

# exec : pod에 접속해 명령어 실행
watch -n 1 kubectl exec liveness-pod -c liveness-container -- ls /tmp/

# liveness-exec container : healthy 라는 파일 만들고, 30초 기다리고, healthy 파일 삭제하고, 1분 기다리고, 종료됨

# livenessProbe : 명령이 성공하면 살아있다, 명령이 실패하면 리스타트 -> 5초마다 healthy 파일 cat, 5번 실패하면 죽은걸로 판단하고 리스타트
```



#### 4. container

**initContainer**

pod에 application container가 실행되기 전에, 사전 작업이 필요한 경우 init container를 통해 작업

*init container가 최종적으로 실행이 되지 않을  시, application container 실행되지 않음*

`vim initcontainer.yaml`

```yaml
apiVersion: v1
kind: Pod
metadata:
	name: init-pod
spec:
	initContainers:
	- name: init-container01
	  image: busybox
	  command: ["/bin/sh", "-c", "sleep 10"]
	- name: init-container02
	  image: busybox
	  command: 
	  - /bin/sh
	  - -c
	  - sleep 10
	containers:
	- name: my-container
	  image: busybox
	  command: ["/bin/sh", "-c", "echo 'hello init container'; sleep 600;"]
```

```bash
# init:0/2 -> running으로 변경되는 것 볼 수 있음
kubectl apply -f initcontainer.yaml

# log 확인
kubectl logs init-pod
```



**resource 할당**

*resources.yaml*

```yaml
apiVersion: v1
kind: Pod
metadata:
	name: resource-pod
spec:
	containers:
	- name: resource-container
	  image: nginx
	  resources:
	  	requests:
	  		cpu: 0.1
	  		memory: 200M
	  	limits:
	  		cpu: 0.5
	  		memory: 1G
```

```bash
kubectl apply -f resources.yaml
# .spec.containers[].resources.requests : 최소한
# .spec.containers[].resources.limits : 최대한

# 단위
# - cpu : 1 (100%), 0.1 (10%)
# - memory : 1 (byte) , 1M (mb), 1G (gb)

vim resources.yaml
# pod, container name 수정 
name: resource-pod-error
# requests, limits 수정
cpu: 10

kubectl apply -f resources.yaml
# status pending에서 넘어가지 못함
# node : <none> 
# -> pod가 어떤 node에서 실행될 지 모른다 
# -> scheduling이 되지 않음 (cpu 10 이라는 요구사항을 만족하는 node가 없음)
# -> cpu 10개인 node를 추가로 join하면 해당 node에 배포될 것
```



**환경변수 설정**

container를 시작할 때 환경변수를 생성(정의)/수정 가능

*vim env-pod.yaml*

```yaml
apiVersion: v1
kind: Pod
metadata:
	name: env-pod
spec:
	containers:
	- name: env-container
	  image: busybox
	  command: ["/bin/sh", "-c", "sleep 6000"]
	  env:
	  - name: TEST
	    value: "test"
	  - name: HOSTNAME
	    valueFrom:
	      fieldRef:
	        fieldPath: spec.nodeName
	  - name: PODNAME
	    valueFrom:
	      fieldRef:
	        fieldPath: metadata.name
```

```bash
kubectl apply -f env-pod.yaml

# node 확인 후 해당노드에서 실행
kubectl exec -it env-pod -- sh
echo $TEST
echo $HOSTNAME
echo $PODNAME
env
# k8s 가 만들어놓은 환경변수와 함께 내가 만든 환경변수 만들어져 있는 것 확인
exit
```



**configmap**

```bash
# configmap 생성 방법
# 1. from-literal 사용하여 생성
kubectl create configmap configmap01 --from-literal=username=dongheon --from-literal=password=1234

# 확인
kubectl get configmap configmap01
# yaml 형태로 확인 (data 2개)
kubectl get configmap configmap01 -o yaml
```

*vim test.properties*

```properties
address=suwon
phone=01012345678
```

```bash
# 2. 파일을 읽어서 yaml 파일 생성
kubectl create configmap configmap02 --from-file=test.properties

# 파일이름이 key가 된다.
kubectl get configmap configmap02 -o yaml
```

*vim using-configmap.yaml*

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: configmap-pod
spec:
  containers:
  - name: configmap-container
    image: busybox
    command: ["/bin/'sh", "-c", "echo $username && echo $password && echo $fileconfig"]
    env:
    - name: username
      valueFrom:
        configMapKeyRef:
          name: configmap01
          key: username
    - name: password
      valueFrom:
        configMapKeyRef:
          name: configmap01
          key: password
    - name: fileconfig
      valueFrom:
        configMapKeyRef:
          name: configmap02
          key: test.properties
```

```bash
kubectl apply -f using-configmap.yaml

# 출력 로그 확인
kubectl logs configmap-pod
```



**secret을 사용해서 환경변수 설정**

*vim secret01.yaml*

```yaml
apiVersion: v1
kind: Secret
metadata:
	name: secret01
type: Opaque
stringData:
	username: dongheon
	password: "1234"
```

*vim env-secret.yaml*

```yaml
apiVersion: v1
kind: Pod
metadata:
	name: secret-pod
spec:
	containers:
	- name: secret-container
	  image: busybox
	  command: ["/bin/sh", "-c", "sleep 6000"]
	  env:
	  - name: myid
	    valueFrom:
	    	secretKeyRef:
	    		name: secret01
	    		key: username
	  - name: mypw
	  	valueFrom:
	  		secretKyeRef:
	  			name: secret01
	  			key: password
```

```bash
kubectl apply -f secret01.yaml
kubectl apply -f env-secret.yaml

# node 확인 후 해당노드에서 실행
kubectl exec -it secret-pod -- sh
echo $myid
echo $mypw
# dongheon, 1234 
exit

# secret 확인
kubectl get secret secret01 -o jsonpath='{.data}'
# base64로 인코딩해서 저장됨

# base64로 인코딩
# -n : echo로 인해 생기는 \n을 제거하고 출력
echo -n dongheon | base64
# ZG9uZ2hlb24=
echo -n 1234 | base64
# MTIzNA==
```



*vim secret02.yaml*

```yaml
apiVersion: v1
kind: Secret
metadata:
	name: secret02
type: Opaque
data:
	username: ZG9uZ2hlb24=
	password: MTIzNA==
```

```bash
kubectl apply -f secret02.yaml

vim env-secret.yaml
# secret01을 secret02로 모두 변경 후 저장
kubectl delete pod secret-pod
kubectl apply -f env-secret.yaml
# node 확인 후 해당노드에서 실행
kubectl exec -it secret-pod -- sh
echo $myid
echo $mypw
# dongheon
# 1234
exit

# secret 확인
kubectl get secret secret02 -o jsonpath='{.data}'

# data field의 value들은 base64로 인코딩되어야 함!
```



**label**

*vim label01.yaml*

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: server01
  labels:
    app: web-server
    env: develop
spec:
  containers:
  - name: nginx
    image: nginx
    ports:
    - containerPort: 80
```

*vim label02.yaml*

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: server02
  labels:
    app: web-server
    env: production
spec:
  containers:
  - name: nginx
    image: nginx
    ports:
    - containerPort: 80
```

```bash
# 한 번에 여러개 배포
kubectl apply -f label01.yaml -f label02.yaml

# label 확인
kubectl get pods --show-labels

# selector
# 등호기반
kubectl get pods --selector env=develop
kubectl get pods -l env=develop
kubectl get pods -l env!=production
# 집합기반
kubectl get pods -l "env in (develop, production)"
```



#### 5. controller

\- replication : 지금은 사용안함 (k8s 처음 만들어졌을 때 부터 사용했었음). 같은 yaml로 만들어진 pod 복제본들(replica)을 관리하는 컨트롤러
\- replicaset : replication controller 발전형. replication은 등호기반 selector만 지원, replicaset은 집합기반 selector까지 지원. 단독으로 사용하는 경우는 거의 없고, deployement controller와 함께 사용
\- deployment : pod와 replicaset 생성 및 관리
\- daemonset : 클러스터 전체 노드에 특정 파드를 하나씩 실행할 때 사용
\- statefulset : 애플리케이션 상태(state)관리
\- job :  작업관리
\- cronjob : cron에 맞춰 작업관리



**replicaset controller**

*vim replica.yaml*

```yaml
apiVersion: apps/v1
kind: ReplicaSet
metadata:
	name: replica-set
spec:
	template:
		metadata:
			name: replica-pod
			labels:
				env: dev
		spec:
			containers:
			- name: replica-container
			  image: nginx
	replicas: 2
	selector:
		matchLabels:
			env: dev
```

**replicaset spec에 반드시 들어가야 하는 내용**

template : (= pod template) pod를 생성하기 위한 설정 
replicas : 복제되는 pod의 갯수 (유지되야 하는 pod의 갯수)
selector : 유지되야 하는 pod를 선택하는 구문

``` bash
kubectl apply -f replica.yaml
# watch -n 1 kubectl get pods,rs -o wide

# 강제로 pod 하나 삭제
kubectl delete pods replica-set-5jst2

# 새로운 pod 다시 생성되어 있는 걸 확인할 수 있다
# rs(replicaset)
kubectl get rs,pods -o wide

# replicaset의 개수 변경
kubectl scale rs replica-set --replicas 3
# pod 3개가 된 것 확인

# 다시 2개로 변경
kubectl scale rs replica-set --replicas 2
# 나중에 만들어진 pod 하나 삭제되는거 확인

# replicaset을 삭제하면 replicaset controller가 관리하던 pod도 모두 삭제됨
kubectl delete rs replica-set

# 다시 생성
kubectl apply -f replica.yaml

# replicaset만 삭제하고 싶을 때
kubectl delete rs replica-set --cascade=orphan
# 확인해보면 replicaset은 없다.
kubectl get rs,pods -o wide
```



**deployment controller**

```bash
# deploy까지 확인하자
watch -n 1 kubectl get deploy,rs,pods -o wide
```

*vim deployment.yaml*

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
	name: deployment
	labels:
		env: dev
spec:
	replicas: 2
	selector:
		matchLabels:
			env: dev
	template:
		metadata:
			labels:
				env: dev
		spec:
			containers:
			- name: deploy-container
			  image: nginx

```

```bash
kubectl apply -f deployment.yaml
# replicaset -> deployment의 hash 정보(yaml)를 가진다.

# pod 하나 삭제해보자
kubectl delete pods deployment-84bccc44bd-8h8ms
# 삭제되었다가 다시 생성되는걸 볼 수 있다.

# scale out/in
kubectl scale deploy deployment --replicas 3
kubectl scale deploy deployment --replicas 2

# 삭제
kubectl delete deploy deployment
```



**rolling update**

\- kubectl set image
\- kubectl edit
\- yaml 수정 -> 재배포 (가장 일반적)

*vim rolling.yaml*

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
 	name: rolling
	annotations:
		kubernetes.io/change-cause: version 1.27.2
spec:
 	replicas: 3
 	selector:
		matchLabels:
			env: dev
	template:
		metadata:
			labels:
				env: dev
		spec:
			containers:
			- name: rolling-container
        	  image: nginx:1.27.2
```

```bash
# nginx 옛날 버전으로 올린다.
# github nginx releases 에서 가장 아래있는 버전으로 가져옴
kubectl apply -f rolling.yaml

# 세가지 방법
# 1. set image deployment deployment이름 컨테이너이름
kubectl set image deployment rolling rolling-container=nginx:1.28.0 

kubectl describe deployment rolling
# pod template의 image가 1.28.0으로 변경된 부분 확인!
```

*rollout 되는걸 볼 수 있다*

old replicaset 에서 new replicaset으로 rolling update (pod가 하나하나 올라가고 내려가면서 downtime 없이 업데이트 됨)

```bash
# 2. kubectl edit
# 여러가지 정보/속성을 수정할 때 사용 가능
kubectl edit deployment rolling
# /tmp/kubectl-edit-t1tip.yaml
# 실행되어있는 환경을 임시파일로 열어줌
# kubernetes.io/change-cause: version 1.29.0
# image: nginx:1.29.0 로 변경
kubectl describe deployment rolling | grep Image
# 1.29.0으로 변경됨!
```

```bash
# 3. yaml 파일 내용 수정 후 적용
vim rolling.yaml
# kubernetes.io.change-cause랑 image 버전 둘 다 1.28.0으로 변경 후 저장
kubectl apply -f rolling.yaml
# 확인
kubectl describe deployment rolling | grep Image
```



**rollbcak**

```bash
# rollout : 배포
kubectl rollout history deployment rolling

# 1.27.2 -> 1.28.0 -> 1.29.0 -> 1.28.0 으로 변경되었으나, yaml의 change-cause를 변경한건 1,28.0밖에 없어서 change cahse가 1.27.2 -> 1.27.2 -> 1.28.0 로 보여진다.
# 3개만 보여진다

# 직전 번호 확인
kubectl rollout history deployment rolling --revision=3

# 직전 상태로 돌아가기
kubectl rollout undo deployment rolling
# 확인 (1.29.0으로 되돌아가 있음)
kubectl describe deployment rolling | grep Image

# history 확인
kubectl rollout history deployment rolling
```



**daemonset controller**

*클러스터 전체 노드에 특정 파드를 하나씩*

*vim daemon.yaml*

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
	name: daemonset
spec:
	selector:
		matchLabels:
			name: daemonset
	updateStrategy:
		type: RollingUpdate
	template:
		metadata:
			labels:
				name: daemonset
		spec:
			containers:
			- name; daemonset
			  image: nginx
```

```bash
watch -n 1 kubectl get ds -o wide

kubectl apply -f daemon.yaml
# master node에는 pod가 배포되지 않아서 2개 (worker1 / worker2)

# taint(node) & toleration(pods)
# taint : 해당 node에 pod 배포 x
# master node에는 taint가 걸려있어서 daemonset으로 인한 pod 배포가 되지 않음
# scheduling 제약사항이 걸려있다.
# key(=value):effect
kubectl describe nodes master | grep -i taint
kubectl describe nodes worker1 | grep -i taint
kubectl describe nodes worker2 | grep -i taint
# masternode는 보이고, worker1/worker2에는 안보임
```

*vim toleration.yaml*

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
	name: daemonset
spec:
	selector:
		matchLabels:
			name: daemonset
	updateStrategy:
		type: RollingUpdate
	template:
		metadata:
			labels:
				name: daemonset
		spec:
			tolerations:
			- key: node-role.kubernetes.io/control-plane
			  effect: NoSchedule
			containers:
			- name; daemonset
			  image: nginx
```

```bash
kubectl delete ds daemonset

# toleration : taint가 있어도 pod 배포
kubectl apply -f toleration.yaml
# 3개 만들어진거 확인
```



**update strategy**

Deployment (update.strategy)
\- Recreate
\- RollingUpdate

Daemonset (updatestrategy)
\- RollingUpdate 
\- OnDelete : pod를 "수동으로" 삭제하면 그 때 업데이트

```bash
watch -n 1 kubectl get ds,pods -o wide

vim daemon.yml

# type: RollingUpdate -> OnDelete

# 적용
kubectl apply -f daemon.yaml 
# 하나 삭제
kubectl delete pod damonset-fcfvs
# up-to-date 하나씩 올라가는거 확인 (worker가 2개라서 2까지밖에 안올라감)
```



**statefulset controller**

\- stateless : 다시 만들어지는 pod가 이전의 pod와 동일하지 않음 (이름 다름)
\- stateful : 이전의 정보를 가지고 다시 만들어지는 pod (이름 동일) -> 식별 관련 정보 (name, 연결정보, ...)

*vim stateful.yaml*

```yaml
apiVersion: v1
kind: Service
metadata:
	name: stateful-service
spec:
	selector:
		env: dev
	ports:
	- port: 80
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
	name: stateful
spec:
	selector:
		matchLabels:
			env: dev
	serviceName: "stateful-service"
	replicas: 3
	template:
		metadata:
			labels:
				env: dev
		spec:
			containers:
			- name: stateful
			  image: nginx
```

```bash
watch -n 1 kubectl get sts,pod -o wide

# -0, -1, -2 생성됨
kubectl apply -f stateful.yaml

# 5개로 올려보자
kubectl scale sts stateful --replicas 5
# 0~4까지 5개 생성

# 3으로 줄이면 큰 숫자부터 순차적으로 죽음
kubectl scale sts stateful --replicas 3

# 삭제
kubectl delete pod stateful-0
# 제거된 pod가 그대로 다시 생성됨 (같은 이름)
```



**job controller**

*vim job.yaml*

```yaml
apiVersion: batch/v1
kind: Job
metadata:
	name: job
spec:
	template:
		spec:
			containers:
			- name: job
			  image: busybox
			  commad: ["/bin/sh", "-c", "echo Hello"]
			restartPolicy: Never
```

```bash
kubectl apply -f job.yaml

# 출력 로그 확인
kubctl logs job-vrtp6
```



**cronjob controller**

`vim cron.yaml`

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
	name: cron
spec:
	schedule: "* * * * *"
	jobTemplate:
		spec:
			template:
				spec:
					containers:
					- name: cron
					  image: busybox
					  command: ["/bin/sh", "-c", "echo Hello"]
					restartPolicy: Never
```

```bash
watch -n 1 kubectl get cronjob,jobs,pods -o wide

# 분(0-59) 시(0-23) 일(1~31) 월(1~12) 요일(1월-6토/0,7일)
# ex) 
# 12월 25일 오전 9시 정각 : 0 9 25 12 *
# 매 주 일요일 새벽 4시 정각0 4 * * 0
# 5분마다 : */5 * * * *

# schedule: "*/1 * * * *"
# 1분마다
# jobTemplate:
# jotTemplate을 만들어라

# 시간이 조금 걸림
kubectl apply -f cron.yaml

# pod가 실행되고, 1분 후 다음 pod가 실행됨
# 확인
kubectl logs cron-29242180-l258d
# hello 뜸


vim cron.yaml

###
jobTemplate:
	spec:
  		completions: 4
  		parallelism: 2
		template:
###

kubectl delete cronjob cron

kubectl apply -f cron.yaml
# 2개 병렬
# 4개의 pod가 completed되어야 job이 완료
# -> 1분에 총 4개가 완료되어야 하는데, 2개씩 병렬로 실행할거다.

kubectl delete cronjob cron
```



#### 6. service

**service 라는 type의 object를 뜻함!**

pod는 일회용 / service는 외부 배포

\- ClusterIP : cluster 내부에서만 사용
\- NodePort : 가장 많이 사용 (ClusterIP + @)
\- LoadBalancer : L4 switch 역할 (public cloud와 service 연결)
\- ExternalName : cluster 내부의 서비스를 외부 dns와 연결
\- headless : pod에 직접 접근 (statefulset과 함께 사용 등)

```bash
watch -n 1 kubectl get svc,deploy,pods -o wide

# stateful controller에서 service 잠깐 써봤음!
kubectl delete svc --all
# service/kubernetes 는 항상 실행됨 (꼭 필요한 service)

kubectl create deployment nginx-service --image=nginx --replicas=2 --port=80

# 확인
curl -sf 10.244.235.162
# 다른 pod 에도 요청
curl -sf 10.244.189.95

# index.html 변경
echo pod 1 > index.html
kubectl cp index.html nginx-service-68f65b99f4-mxk4h:/usr/share/nginx/html/index.html

echo pod 2 > index.html
kubectl cp index.html nginx-service-68f65b99f4-g2h68:/usr/share/nginx/html/index.html

# 확인
curl -sf 10.244.235.162
# 다른 pod 에도 요청
curl -sf 10.244.189.95

# service ip에 요청
curl -sf 10.96.0.1
# 현재는 요청 안먹힘
```

*vim cluster-ip.yaml*

```yaml
apiVersion: v1
kind: Service
metadata:
	name: cluster-ip
spec:
	type: ClusterIP
	selector:
		app: nginx-service
	ports:
	- protocol: TCP
	  port:80
	  targetPort: 80
```

```bash
# 실행
kubectl apply -f cluster-ip.yaml

# cluster ip에 요청 (여러번 해보기)
curl -sf 10.108.20.154
# 1이랑 2가 랜덤으로 응답

# pod 하나 삭제하고 다시 요청 (삭제하면 다시 만들어짐)
sudo kubectl delete pod nginx-for-svc-68f65b99f4-g2h68

curl -sf 10.108.20.154
# 새로 만들어진 pod는 thank you for ~
```



*vim nodeport.yaml*

```yaml
apiVersion: v1
kind: Service
metadata:
	name: nodeport
spec:
	type: NodePort
	selector:
		app: nginx-service
	ports:
	- protocol: TCP
	  port: 80
	  targetPort: 80
	  nodePort: 30000
```

```bash
# nodePort: 지정하지 않으면 30000-32767 범위 내에서 임의의 포트 자동 할당
kubectl apply -f nodeport.yaml

# worker ip로 접속해보기 (외부 ip 접속)
curl -sf 192.168.98.131:30000
```



*vim loadbalancer.yaml*

```yaml
apiVersion: v1
kind: Service
metadata:
  name: loadbalancer
spec:
  type: LoadBalancer
  selector:
    app: nginx-service
  ports:
  - protocol: TCP
    port: 80
    targetPort: 80
```

```bash
kubectl apply -f loadbalancer.yaml

# external-ip가 pending에서 넘어가지지 않는다.
# local환경에서는 loadbalancer 기능이 지원되지 않기 때문!
# 클라우드 환경 (aws, gcp, azure 등등)에서 해당 기능이 필요할 때 사용!
```



**externalname**

*vim external.yaml*

```yaml
apiVersion: v1
kind: Service
metadata:
	name: externalname
spec:
	type: ExternalName
	externalName: google.com
```

```bash
kubectl apply -f external.yaml

# alpine으로 domain name 확인
# --rm : 일 끝나면 삭제해라
kubectl run -it --rm alpine --image=alpine -- /bin/sh
# curl 설치
apk add curl
# ip 대신 domain name으로 접근할 수 있다.
# 이름.네임스페이스.타입.클러스터.로컬
curl -sf externalname.default.svc.cluster.local
# www.google.com css가 보임!
exit
```



**headless service**

```bash
# headless service : ip가 없는 service
# domain name으로 연결

# 기존 service 삭제 
kubectl delete svc --all
# nginx-service는 안죽었다. (이름만 service지 deployment로 만들어서)
```

*vim headless.yaml*

```yaml
apiVersion: v1
kind: Service
metadata:
	name: headless
spec:
	clusterIP: None
	selector:
		app: nginx-service
	ports:
	- protocol: TCP
	  port: 80
	  targetPort: 80
```

```bash
kubectl apply -f headless.yaml

kubectl run -it --rm alpine --image=alpine -- /bin/sh

apk add curl

curl -sf headless.default.svc.cluster.local
exit
```



#### 7. ingress

cluster 외부에서 받은 요청을 어떻게 처리할지 정의해둔 "규칙" (명세)
\- L7 (application layer) 역할
\- process가 아닌 명세
\- 실제로 동작하는 것은 ingress controller (pod) 
\- 공식적인 ingress controller : ingress-gce (google) / ingress-nginx 

```bash
# 이전 실습내용 삭제
kubectl delete svc,deploy,pod --all

watch -n 1 kubectl get ing,svc,deploy,pods -o wide

# pod 생성
kubectl create deploy service01 --image nginx --port 80
kubectl create deploy service02 --image nginx --port 80

# 식별 편하도록 pod 내용 수정
echo service01 > index.html
kubectl cp index.html service01-9749c66f-ppxz5:/usr/share/nginx/html/index.html

echo service02 > index.html
kubectl cp index.html service02-76559b6f5d-zhj7d:/usr/share/nginx/html/index.html

# service 생성
kubectl expose deploy service01
kubectl expose deploy service02

# 확인
curl -sf 10.244.235.173
curl -sf 10.244.189.106

# nginx ingress controller 설치
# github kubernetes nginx -> deploy -> static/provider -> baremetal -> deploy.yaml -> raw
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/refs/heads/main/deploy/static/provider/baremetal/deploy.yaml
# aws : aws
# cloud : azure, gcp, ...
# baremetal : local

kubectl get ns
# ingress controller는 별도의 namespace로 만들어짐
# ingress-nginx
```

*vim ingress-config.yaml*

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
	name: service-ingress
	annotations:
		nginx.ingress.kubernetes.io/rewrite-target: /
spec:
	ingressClassName: nginx
	rules:
	- host: service.com
	  http:
		paths:
		- path: /service01
		  pathType: Prefix
		  backend:
		  	service:
		  		name: service01
		  		port:
		  			number: 80
		- path: /service02
		  pathType: Prefix
		  backend:
		  	service:
		  		name: service02
		  		port:
		  			number: 80
```

```bash
kubectl apply -f ingress-config.yaml

kubectl describe ingress service-ingress

# ingress address로 추가
sudo vim /etc/hosts
192.168.98.131 service.com

kubectl get svc,deploy -n ingress-nginx
# pod (ingress-nginx-controller)
# ingress (rule) 를 가지고 실제 처리해줄 container
# service (ingress-nginx-controller / NodePort / 10.108.14.202 / 80:30241)

# 확인
curl -sf service.com:30294/service01
curl -sf service.com:30294/service02

# 모두 삭제 방법
kubectl delete ing,svc,deploy,pod --all

kubectl delete -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/refs/heads/main/deploy/static/provider/baremetal/deploy.yaml

# 현재 실행중인 모든 것 확인
kubectl get all
```



#### 8. volume

영구적 데이터 저장 공간

\- hostPath : 가장 저렴하게 구축 가능
\- emptyDir : 영구적으로 유지되진 않지만, 공유되게 만들고 싶을 때 - local disk
\- public cloud vendor가 제공하는 storage service 들



**empty dir**

*vim empty-volume.yaml*

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: emptydir
spec:
  containers:
  - name: insert-container
    image: busybox
    command: ["/bin/sh", "-c", "echo '<h1>hello</h1>' > /html/index.html; sleep 6000;"]
    volumeMounts:
    - name: emptyvolume
      mountPath: /html
  - name: nginx-container
    image: nginx
    ports:
    - containerPort: 80
    volumeMounts:
    - name: emptyvolume
      mountPath: /usr/share/nginx/html
  volumes:
  - name: emptyvolume
    emptyDir: {}
```

```bash
kubectl apply -f empty-volume.yaml

# 확인
curl 10.244.189.119
# pod uid 확인
kubectl get pods emptydir -o yaml | grep uid

# pod가 배포된 해당 worker로 가서 su 비밀번호 설정
sudo passwd
# su
su
cd /var/lib/kubelet/pods/
# 위에서 확인한 uid 접속
cd 9139c904-b970-408d-ac8b-cbac4b435645
ls
# container와 volume 확인 가능
cd volumes
cd kubernetes.io-empty-dir
cd emptyvolume
cat index.html
```



**hostpath**

*hostpath의 type 종류*

\- DirectorOrCreate
\- Directory
\- FileOrCreate
\- File
\- Socket
\- CharDevice
\- BlockDevice

*vim host-volume.yaml*

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: hostpath
spec:
  containers:
  - name: nginx-container
    image: nginx
    ports:
    - containerPort: 80
    volumeMounts:
    - name: hostvolume
      mountPath: /usr/share/nginx/html
  volumes:
  - name: hostvolume
    hostPath:
      path: /html
      type: DirectoryOrCreate
```

```bash
kubectl apply -f host-volume.yaml

# pod가 배포된 worker에 가서
cd /html
sudo vim index.html

<h1>hostpath</h1>
# :wq!

# master 가서 확인
curl 10.244.189.121
# pod 삭제
kubectl delete pod hostpath
# 그래도 해당 worker에는 /html 남아있음
# (emptydir은 pod 안에 생성되기때문에, pod 삭제하면 사라짐
# 다시 배포하면 연결이 될까?
kubectl apply -f host-volume.yaml
curl 10.244.189.122
# 단점! 만일 다른 node에 배포되면 이전 데이터 확인 불가
```



**nfs**

Network FileSystem

hostpath는 같은 node에 올라가야 이전 데이터를 확인 가능하다는 단점이 있음
-> shared volume 이 필요

여러 개 pod가 하나의 volume에 공유해 읽기/쓰기 가능!

```bash
# nfs server 설치 - pod가 배포될 가능성이 있는 모든 node에 설치
sudo apt install nfs-common -y

# master 에서
sudo apt install nfs-kernel-server
# 공유디렉토리 생성
sudo mkdir -p /data
# 권한 설정
sudo chmod 777 /data
# 공유 설정 추가
echo "/data *(rw,sync,no_subtree_check)" | sudo tee -a /etc/exports
# nfs 서버 재시작
sudo systemctl restart nfs-server
```

\- PersistentVolume : volume. 이전과 다른 형태로 구성되어 있을 뿐 다른 것 없음 (이전엔 pod와 volume이 같이 배포되었고, 지금은 따로 배포)
\- PersistentVolumeClaim : pv에 요청

*pod container와 volume을 따로 배포하자*
\- pod container : developer
\- volume : server engineer, storage engineer, infra 담당자, ...

*vim nfs-pv.yaml*

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: nfs-pv
spec:
  capacity:
    storage: 1Gi
  accessModes:
    - ReadWriteMany  # 여러 파드에서 읽기/쓰기 가능
  nfs:
    path: /data  # 마스터 노드의 공유 폴더 경로
    server: 192.168.98.129 # 마스터 노드의 IP 주소
```

*vim nfs-pvc.yaml*

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: nfs-pvc
spec:
  accessModes:
    - ReadWriteMany  # PV와 동일한 접근 모드
  resources:
    requests:
      storage: 1Gi  # PV 용량과 일치하거나 작아야 함
```

```bash
kubectl apply -f nfs-pv.yaml
kubectl apply -f nfs-pvc.yaml

watch -n 1 kubectl get pv,pvc,pods -o wide
```

*vim nfs-nginx.yaml*

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: nfs-nginx
spec:
  containers:
    - name: nfs-container
      image: nginx
      ports:
        - containerPort: 80
      volumeMounts:
        - name: nfsvolume
          mountPath: /usr/share/nginx/html
  volumes:
    - name: nfsvolume
      persistentVolumeClaim:
        claimName: nfs-pvc # 이전에 생성한 PVC의 이름
```

```bash
kubectl apply -f nfs-nginx.yaml

cd /data
vim index.html

<h1>nfs</h1>
# :wq!

cd
# nf-nginx pod 접속
curl 10.244.189.123
# /data 안의 index.html이 공유된걸 알 수 있음
```





#### 9. configmap volume mount

*test.properties*

```properties
address=suwon
phone=01012345678
```

```bash
# 파일을 읽어서 yaml 파일 생성 (한적 있음)
kubectl create configmap configmap02 --from-file=test.properties

# 파일이름이 key가 된다.
kubectl get configmap configmap02 -o yaml
```

*vim volume-config.yaml*

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: volume-confi-pod
spec:
  containers:
  - name: volume-config-container
    image: busybox
    command: ["/bin/sh", "-c", " cat /config/test.properties && sleep 6000"]
    volumeMounts:
    - name: config-volume
      mountPath: /config
  volumes:
  - name: config-volume
    configMap:
      name: configmap02
```

```bash
kubectl apply -f volume-config.yaml

kubectl logs volume-config-pod
```




