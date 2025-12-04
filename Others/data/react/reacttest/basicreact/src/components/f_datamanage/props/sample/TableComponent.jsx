import React from "react";

export default function TableComponent({ header = [], body = [] }) {
  return (
    <table>
      <thead>
        <tr>
          {header.length > 0 && header.map((h, i) => <th key={i}>{h}</th>)}
        </tr>
      </thead>
      <tbody>
        {body.length > 0 &&
          body.map((data) => {
            return (
              <tr key={data.no}>
                {Object.values(data).map((v, i) => (
                  <td key={i}>
                    {typeof v == "object"
                      ? `${v.getFullYear()}-${(v.getMonth() + 1)
                          .toString()
                          .padStart(2, "0")}-${v
                          .getDate()
                          .toString()
                          .padStart(2, "0")}`
                      : v}
                  </td>
                ))}
              </tr>
            );
          })}
      </tbody>
    </table>
  );
}
