import React, { useState } from "react";

import {
  eventHandler,
  highFuncHanler,
  debouncer,
  throttle,
} from "../../lib/event/Handlers";

const outerHandler = (event) => {
  alert("외부 핸들러 호출");
};
//state데이터 이용하기
const notAccessStateHandler = () => {
  //외부에 선언되어있어 접근이 불가능함.
  console.log(testData);
};
const outerUseStateHandler = (stateData) => (e) => {
  console.log(stateData);
};

export default function B_FunctionEvent() {
  //데이터 이용하기
  const [testData, setTestData] = useState("기본값");

  const innerHandler = () => {
    //내부에 선언되어 있어 접근이 가능함.
    console.log(testData);
    alert("내부핸들러 호출");
  };

  const useStateHandler = () => {
    //내부에 선언되어 있어 접근이 가능함.
    console.log(testData);
  };
  const changeState = (e) => {
    setTestData(e.target.value);
  };
  return (
    <div>
      <h2>선언된 함수를 핸들러로 이용하기</h2>
      <p>
        이벤트 속성에 함수를 함수명만 지정, 함수를 호출하는 구문을 작성하지 않음
      </p>
      <p>선언위치</p>
      <ul>
        <li>
          컴포넌트함수 내부 : 컴포넌트 함수 내부에 핸들러를 선언함. / 페이지가
          랜더링될때 마다 함수가 재선언되는 문제가 있음 / state, props값에
          접근할 수 있음
        </li>
        <li>
          컴포넌트함수 외부 : 컴포넌트함수 외부에 핸들러를 선언함. / 페이지가
          랜더링되도 재선언되지 않음 / state, props에 접근하지 못해, 사용시
          매개변수로 전달해야함.
        </li>
      </ul>
      <h3>내부함수 핸들러 등록하기</h3>
      <button onClick={innerHandler}>내부함수호출</button>
      <h3>외부함수 핸들러 등록하기</h3>
      <button onClick={outerHandler}>외부함수호출</button>
      <input onChange={eventHandler} />
      <h3>함수를 호출하는 구문을 설정하면 이벤트가 동작하지 않음</h3>
      <p>
        랜더링시 한번 호출하고 이벤트 발생시 동작하지 않음. 주의 고차함수로
        선언한 함수는 가능함 * alert을 두번호출
      </p>
      {/* <button onClick={outerHandler()}>함수호출구문을 설정</button> */}
      <p>
        이벤트 실행시 특정값이 필요할 경우 고차함수로 이벤트핸들러를 설정할 수
        있음 특정값 : state, props값이나 state수정하는 함수를 전달할때 사용
        *props, state배우고 해보자.
      </p>
      <button onClick={highFuncHanler("test")}>고차함수로 설정</button>
      <h3>간단하게 state값 이용하기</h3>
      <p>
        state는 반응성(변경할 수 있는)갖는 데이터로 유동적인 값을 사용할때
        React에서 사용
      </p>
      <button onClick={useStateHandler}>state값 가져오기</button>&nbsp;
      <button onClick={notAccessStateHandler}>state값 못 가져오기</button>
      <p>고차함수를 이용하면 state값을 이용할 수 있음</p>
      <button onClick={outerUseStateHandler(testData)}>이용하기</button>
      <p>state값 수정하기</p>
      <input type="text" onChange={changeState} />
      <h3>Debouncing 이용하기</h3>
      <p>
        지속적으로 발생하는 이벤트를 모두 실행하지 않고 특정시간(delaytime)이
        지난 후 마지막에 한개만 실행하게 하는 기술 keyup, click이벤트에 적용할
        수 있음
      </p>
      <input
        type="text"
        onChange={debouncer(([e]) => {
          e.target.nextElementSibling.innerText = e.target.value;
        }, 300)}
      />
      <span></span>
      <h4>버튼에 적용하기</h4>
      <p>클릭할때마다 alert을 실행하지 않고 한번만 실행</p>
      <button
        onClick={debouncer(([e]) => {
          console.log("클릭함");
        })}
      >
        클릭하면 console출력-debouncer
      </button>
      <button
        onClick={() => {
          console.log("클릭함");
        }}
      >
        클릭하면 console출력
      </button>
      <h3>Throttling 이용하기</h3>
      <p>
        특정시간를 기준으로 주기적으로 실행하는 것 무한스크롤을 처리할때 많이
        사용
      </p>
      <h4>스크롤에 적용하기</h4>
      <div
        id="throttleTest"
        style={{ height: "100px", overflow: "auto" }}
        onScroll={throttle(() => {
          console.log("실행"); //2초에 한번씩 실행이 출력됨.
        })}
      >
        UI & 그래픽 처리 🖥️ 사용자의 시각적 경험과 직접적으로 관련된 부드러운
        인터랙션을 만드는 데 효과적입니다. 창 크기 조절 (resize 이벤트):
        브라우저 창 크기를 조절할 때 resize 이벤트는 수십, 수백 번씩 발생합니다.
        이때마다 복잡한 레이아웃 계산을 다시 하면 브라우저가 버벅입니다.
        쓰로틀링을 적용하면 0.1초에 한 번씩만 계산하도록 만들어 부드러운
        리사이즈 경험을 제공할 수 있습니다. 마우스 움직임 추적 (mousemove
        이벤트): 마우스 커서를 따라다니는 애니메이션이나 특정 영역에 들어왔는지
        감지할 때 사용됩니다. 마우스가 1px 움직일 때마다 이벤트를 처리하는 것은
        낭비이므로, 쓰로틀링으로 일정 간격으로만 위치를 업데이트하여
        애니메이션을 부드럽게 만들고 CPU 사용량을 줄입니다.
      </div>
    </div>
  );
}
