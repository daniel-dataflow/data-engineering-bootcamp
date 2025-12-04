import React from "react";
//외부 데이터 가져와서 화면 구현하기
//data폴더에 저장된 exportData.js파일에 내용을 불러와서 처리하기
//import예약어를 이용해서 불러와 처리
import {
  dataExport,
  funcExport,
  test,
  letDataExport,
  members,
} from "../../data/exportData";

export default function E_ExportData() {
  //외부에서 가져온 데이터를 직접수정할 수는 없다.
  // dataExport = "다른 데이터 입력";
  // letDataExport = "이건되니?";
  //수정하려면 복사 후 사용
  let testVal = dataExport;
  testVal += "천재";
  return (
    <div>
      <h3>외부 js파일로 저장된 데이터 가져오기</h3>
      <ul>
        <li>dataExport : {dataExport}</li>
        <li>
          funcExport : {funcExport()}
          &nbsp;type : {typeof funcExport}
        </li>
        <li>letDataExport : {letDataExport}</li>
        <li>선언적 함수 test : {test()}</li>
      </ul>
      <h4>외부에서 가져온 리스트 데이터출력</h4>
      <table>
        <thead>
          <tr>
            <th>회원번호</th>
            <th>회원아이디</th>
            <th>회원이름</th>
            <th>나이</th>
          </tr>
        </thead>
        <tbody>
          {members.map((v) => (
            <tr key={v.userId}>
              <td>{v.userNo}</td>
              <td>{v.userId}</td>
              <td>{v.userName}</td>
              <td>{v.age}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <h4>변수에 저장하고 값 수정하기</h4>
      <h4>복사 데이터 : {testVal}</h4>
    </div>
  );
}
