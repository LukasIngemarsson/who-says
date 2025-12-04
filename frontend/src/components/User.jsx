import { Avatar, Box, Typography } from "@mui/material";

const User = ({ name }) => {
  return (
    <Box display="flex" alignItems="center" marginY={1}>
      <Avatar sx={{ bgcolor: "primary.main", marginRight: 1 }}>
        {name?.charAt(0).toUpperCase()}
      </Avatar>
      <Typography variant="body1">{name}</Typography>
    </Box>
  );
};

export default User;
