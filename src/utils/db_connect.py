import _mssql
import time
ROOM = '201'
BUILDING = 'B4'

class ConnectDB:
    def __init__(self):
        server = 'bankingdb-hcmut.database.windows.net'
        database = 'FaceCheckingSystem'
        username = 'Bankowner@bankingdb-hcmut'
        password = 'Test1234'   

        self.conn = _mssql.connect(server=server, user=username, password=password, database=database)
    
    def goIn(self, user_id):
        sqlstm = """DECLARE @Check INTEGER;
                    EXEC @Check = FCS.GO_IN
                        @Cus_ID = {},
                        @Building_ID = {},
                        @Room_ID = {};
                    SELECT @Check;"""
        return self.conn.execute_scalar(sqlstm.format(user_id, BUILDING, ROOM))
    
    def goOut(self, user_id ):
        sqlstm = """DECLARE @Check INTEGER;
                    EXEC @Check = FCS.GO_OUT
                        @Cus_ID = {},
                        @Building_ID = {},
                        @Room_ID = {};
                    SELECT @Check;"""
        return self.conn.execute_scalar(sqlstm.format(user_id, BUILDING, ROOM))   

    def close(self):
        self.conn.close()

if __name__ == "__main__":
    db = ConnectDB()
    print(db.goIn(1))
    time.sleep(0.01)
    print(db.goOut(1))
    time.sleep(0.01)
    print(db.goIn(1))
    time.sleep(0.01)
    print(db.goOut(1))
    db.close()