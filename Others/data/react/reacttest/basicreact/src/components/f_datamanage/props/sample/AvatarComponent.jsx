import React from "react";

export default function AvatarComponent({ src, alt, size = 50 }) {
  const style = {
    width: `${size}px`,
    height: `${size}px`,
  };
  const avatarStyle = `rounded-full object-cover`;
  return (
    <>
      <img src={src} alt={alt} style={style} className={avatarStyle} />
    </>
  );
}
