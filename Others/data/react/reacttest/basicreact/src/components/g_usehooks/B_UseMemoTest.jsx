import React from "react";
import B_NonUseMemoCom from "./child/B_NonUseMemoCom";
import B_UseMemoCom from "./child/B_UseMemoCom";
import B_DataFilterCom from "./child/B_DataFilterCom";
import B_ParentComponent from "./child/B_ParentComponent";
export default function B_UseMemoTest() {
  // useMemo() 이용하기
  // 비용이 큰 계산 결과를 기억해두고 의존성이 바뀔때만 다시 계산해주는 Hook
  // memoizedValue : 이전 결과를 기억하는 것
  //
  return (
    <div>
      <h3>useMemo함수이용하기</h3>
      <p>
        랜더링최적화를 하는 Hook, 연산에 비용이 많이 드는경우 계속 실행하지 않고
        기억된 값을 가져와 처리하게 만듦
      </p>
      <h4>성능비교하기</h4>
      <div>
        <h4>메모를 사용하지 않은 컴포넌트</h4>
        <B_NonUseMemoCom></B_NonUseMemoCom>
        <h4>메모를 사용한 포넌트</h4>
        <B_UseMemoCom></B_UseMemoCom>
      </div>
      <h4>필터/정렬된 데이터 useMemo로 관리</h4>
      <B_DataFilterCom />
      <h4>자식 컴포넌트 랜더링 보호하기 -> React.memo()이용하기</h4>
      <p>부모컴포넌트가 랜더링될때 자식도 같이 랜더링 되버려 비효율적임</p>
      <B_ParentComponent></B_ParentComponent>
    </div>
  );
}
