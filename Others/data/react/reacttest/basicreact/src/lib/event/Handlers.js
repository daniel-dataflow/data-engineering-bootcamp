export const eventHandler = (e) => {
  console.log(e.target.value);
};

//고차함수 선언하기
export const highFuncHanler = (data) => (e) => {
  console.log(`data : ${data}`);
  console.log(e.target);
};

//debouncer wrapper만들기
//시간간격을 기준으로 마지막에 한번만 실행하는 것
export const debouncer = (actor, wait = 500) => {
  let delay;
  return (...arg) => {
    if (delay) window.clearTimeout(delay);
    delay = window.setTimeout(() => {
      actor(arg);
    }, wait);
  };
};
//시간간격을 기준으로 주기적으로 실행하는 것 -> 꾸준하게 실행
export const throttle = (action, interval = 2000) => {
  let timecheck;
  return (...args) => {
    if (timecheck) return;
    //쿨타임이 아니면 함수를 실행
    action(...args);
    //쿨타임설정
    timecheck = setTimeout(() => {
      timecheck = null;
    }, interval);
  };
};
