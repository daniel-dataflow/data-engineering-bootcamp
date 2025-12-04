import React, { useState, useEffect } from "react";

export default function D_EffectDebouncing() {
  const [query, setQuery] = useState("");
  const [productList, setProductList] = useState([]);

  //query값이 변경되었을때 데이터를 가져오는 설정
  useEffect(() => {
    //최초실행을 차단하는 조건문
    if (query.trim().length > 0) {
      const request = setTimeout(async () => {
        const response = await fetch(
          `https://dummyjson.com/products/search/?q=${query}`
        );
        const data = await response.json();

        //필요항목만 객체배열로 만들기
        const myproduct = data.products.reduce((prev, next) => {
          const {
            id,
            title,
            category,
            price,
            rating,
            stock,
            brand,
            weight,
            images,
            thumbnail,
          } = next;

          prev.push({
            id,
            title,
            category,
            price,
            rating,
            stock,
            brand,
            weight,
            images,
            thumbnail,
          });
          return prev;
        }, []);
        setProductList(myproduct);
      }, 500);
      //클린업 함수설정 -> userEffect실행시 이전 타임아웃제거
      return () => {
        clearTimeout(request);
      };
    }
  }, [query]);
  //가져온 상품의 내용에 따라 출력할 태그를 만들어주는 함수
  const makeContent = (content) => {
    const reg = /.(png|jpg|jpeg|webp)$/i;
    if (Array.isArray(content)) {
      return content.map((c) => {
        if (reg.test(c.toString())) {
          return <img src={c.toString()} width="100" height="100" />;
        } else {
          return Object.values(c).map((t) => <p>{t}</p>);
        }
      });
      //   });
    } else if (typeof content == "object") {
      return Object.values(content).map((p) => <span>{p}</span>);
    } else {
      if (reg.test(content)) {
        return <img src={content} width="100" height="100" />;
      } else return <span>{content}</span>;
    }
  };
  return (
    <div>
      <h3>조회하기</h3>
      <input
        type="text"
        value={query}
        onChange={(e) => {
          setQuery(e.target.value);
        }}
      />
      <div>
        {productList.length > 0 && (
          <table>
            <thead>
              <tr>
                {Object.keys(productList[0]).map((column, i) => (
                  <th key={i}>{column}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {productList.map((product, i) => (
                <tr key={i}>
                  {Object.values(product).map((p, i) => (
                    <td key={i}>{makeContent(p)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
