import React, { useEffect, useRef, useState } from "react";
// 차트구성 모듈 불러오기
import {
  Chart,
  BarElement,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend,
  LineController,
  LineElement,
  PointElement,
  BarController,
} from "chart.js";

export default function A_UserRefTest() {
  // useRef기본설정예제
  const container = useRef();
  // useRef에서 반환한 HTMLElement객체 정보확인
  const testUserRef = () => {
    //current로 접근해서 Element요소의 속성에 접근할 수 있음
    //userRef().current == document.getElementBy..()함수호출한것과 동일 -> HTMLElement객체가 반환됨.
    console.log(container.current.tagName);
    console.log(container.current.innerText);
  };
  //HTMLElement와 연결하여 데이터 처리하기
  // 포커스 설정하기
  const focusTag = useRef();
  const handleFocus = () => {
    if (focusTag.current) focusTag.current.focus();
  };
  // 스크롤 처리 -> 스크롤을 움직여 원하는 위치로 이동하기
  const bottomRef = useRef(null);
  const [message, setMessage] = useState([]);
  const addMeesage = () => {
    setMessage((pre) => [...pre, `신규메세지 ${pre.length}+1`]);
  };
  useEffect(() => {
    if (bottomRef.current) {
      // scrollIntoView() : 지정한 요소가 화면에 보이게 스크롤을 옮겨주는 함수
      // behavior : "smooth(부드럽게스크롤이 움직임)", "auto(바로 움직임)"
      // block ; 출력할 요소의 위치, start(상단), end(하단), center(중앙), nearest(가까운위치)
      bottomRef.current.scrollIntoView({
        behavior: "smooth",
      });
    }
  }, [message]);

  // 비랜더링 데이터를 활용하는 방식
  // 타이머 설정 -> setInterval ID값을 저장하고 주기적실행을 컨트롤하기
  const [count, setCount] = useState(0);
  const timerIdRef = useRef(null); // 타이머 ID 저장용
  //   let timeId = null; 일반 지역변수는 한번 선언되고 끝나서 사용할 수 없음
  const start = () => {
    if (timerIdRef.current) return; // 이미 타이머가 있으면 중복 생성 방지
    // if (timeId) return;

    const id = setInterval(() => {
      // 이 안에서는 최신 state 보장을 위해 콜백 형태 사용
      setCount((prev) => prev + 1);
    }, 1000);
    timerIdRef.current = id;
    // timeId = id;
  };

  const stop = () => {
    if (timerIdRef.current) {
      clearInterval(timerIdRef.current);
      timerIdRef.current = null; // 타이머 종료 후 ref 값 초기화
      //   timeId = null;
    }
  };

  // 사용자가 입력한 이전 값을 저장하고 여러 함수에서 사용하기
  const [inputData, setInputData] = useState("");
  const preValues = useRef([]);

  const inputDataChange = (e) => {
    setInputData(e.target.value);
  };

  const prevInputDataCheck = () => {
    console.log(preValues.current);
  };

  useEffect(() => {
    if (inputData != "") preValues.current.push(inputData);
    //히스토리는 10개까지만 저장하기 -> 10개가 넘으면 앞에서 부터 삭제
    if (preValues.current.length > 10) preValues.current.shift();
  }, [inputData]);

  //추가 -> 배열로 이전 입력 데이터 관리하기
  const [prevHistory, setPrevHistory] = useState([]);
  const prevDataCheck = () => {
    setPrevHistory(preValues.current);
  };

  // 차트출력하기
  // chart.js패키지를 이용해서 차트를 출력할때 차트를 만드는 객체는
  // state(리랜더링으로 오류발생)가 아닌 useRef로 저장해서 관리해야함.

  const canvasRef = useRef(null);
  const chartRef = useRef(null);

  //char에 출력할 라벨과 데이터
  const [labels, setLebels] = useState(["1월", "2월", "3월", "4월"]);
  const [chartData, setChartData] = useState([120, 150, 180, 90]);
  // 사용하는 요소/플러그인을 char에 등록하고 사용해야함.
  Chart.register(
    BarController,
    LineController,
    BarElement,
    LineElement,
    PointElement,
    CategoryScale,
    LinearScale,
    Tooltip,
    Legend
  );
  useEffect(() => {
    if (chartRef.current) {
      //이전 그래프를 삭제하기 -> 데이터 변경시 차트에 반영
      chartRef.current.destroy();
    }
    //convas태그에서 getContext()를 이용해서 그림 도구를 2d방식으로 가져옴
    const ctx = canvasRef.current.getContext("2d");
    //chart객체 만들기
    chartRef.current = new Chart(ctx, {
      type: "bar", // bar, line, pie, doughnut 등
      data: {
        labels, // x축 라벨 배열
        datasets: [
          {
            label: "월별 매출",
            data: chartData, // y축 데이터 배열
            // backgroundColor 등 스타일도 지정 가능
            backgroundColor: "rgba(75, 192, 192, 0.4)",
            borderColor: "rgba(75, 192, 192, 1)",
            borderWidth: 1,
          },
        ],
      },
      options: {
        responsive: true, // 반응형
        plugins: {
          legend: {
            display: true,
            position: "top",
          },
          tooltip: {
            enabled: true,
          },
        },
        scales: {
          y: {
            beginAtZero: true, // y축 0부터 시작
          },
        },
      },
    });
    //클린업 객체
    return () => {
      if (chartRef.current) {
        chartRef.current.destroy();
      }
    };
  }, [labels, chartData]);
  //추가 기능 넣기
  // 월 데이터 수정시 chart에 바로 반영하기
  const [month, setMonth] = useState();
  const [value, setValue] = useState();
  const monthChoice = (e) => {
    setMonth(parseInt(e.target.value));
  };
  const dataChange = (e) => {
    setValue(e.target.value);
  };
  const changeChartData = (e) => {
    console.log(month + " " + value);
    setChartData((prev) => {
      return prev.map((v, i) => (i == month ? value : v));
    });
  };
  return (
    <div>
      <h4>useRef이용하기</h4>
      <p>
        DOM 요소 접근할 수 있는 hooks으로 DOM 요소의 위치정보나 이전 값 등을
        가져올때 사용 랜더링과 무관한 값을 저장하고 값이 변경되어도 컴포넌트를
        랜더링시키지 않음 매개변수로 초기값을 받음 활용 변수에 useRef()반환값을
        저장하고 DOM정보를 가져올 태그의 ref속성을 설정
      </p>
      <h4>기본 설정하기</h4>
      <div ref={container}>container조작</div>
      <button onClick={testUserRef}>useRefTest</button>

      <h4>특정 태그에 포커스 설정하기</h4>
      <div>
        <input
          type="text"
          ref={focusTag}
          placeholder="버튼을 누르면 여기에 포커스가 와요"
        />
        <button onClick={handleFocus}>포커스설정</button>
      </div>

      <h4>scroll 데이터 처리하기</h4>
      <p>
        새로운 데이터가 추가되면 스크롤을 맨아래로 설정할때 사용, 무한스크롤이나
        알림센터, 채팅에서 많이 사용
      </p>

      <div
        style={{
          border: "1px solid gray",
          height: 150,
          overflowY: "auto",
          padding: 8,
        }}
      >
        {message.map((e, i) => (
          <div key={i}>{e}</div>
        ))}
        <div ref={bottomRef}></div>
      </div>
      <button onClick={addMeesage}>메세지 추가</button>

      <h4>타이머 데이터 처리하기</h4>
      <p>렌더링과 관계없이 시간정보를 저장하고 싶을때 사용할 수 있음</p>
      <div>
        <p>타이머: {count}초</p>
        <button onClick={start}>시작</button>
        <button onClick={stop}>정지</button>
      </div>

      <h4>이전값 저장하기</h4>
      <p>컴포넌트 내부에서 공용으로 이전 값에 대해 처리할때 사용</p>
      <div>
        <input type="text" onInput={inputDataChange} />
        <button onClick={prevInputDataCheck}>이전입력 값들 확인하기</button>
        <button onClick={prevDataCheck}>이전 입력값확인</button>
        <ol>
          {prevHistory.map((v, i) => {
            return <li key={i}>{v}</li>;
          })}
        </ol>
      </div>
      <h4>chart.js이용해서 차트를 출력할때 useRef를 사용함</h4>
      <p>
        차트를 생성하는 객체를 state에 넣으면 오류가 발생하거나 무한루프에 빠질
        수 있어 주의 chart.js설치 : npm install chart.js
      </p>
      <div style={{ width: "600px", height: "400px" }}>
        <canvas ref={canvasRef} />
      </div>
      <select onChange={monthChoice}>
        {labels.map((v, i) => {
          return (
            <option key={i} value={i}>
              {v}
            </option>
          );
        })}
      </select>
      <input type="text" onChange={dataChange} />
      <button onClick={changeChartData}>데이터 적용하기</button>
    </div>
  );
}
