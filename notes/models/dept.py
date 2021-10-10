from models import db

class Dept(db.Model):
    __tablename__ = 't_dept'
    id = db.Column('d_id', db.Integer, primary_key=True)
    d_name = db.Column(db.String(50))
    d_address = db.Column(db.String(100))
    def __init__(self, id, name, addr):
        self.id = id
        self.d_name=name
        self.d_address=addr

    def __str__(self):
        return "(%s, %s, %s)" % (self.id, self.d_name, self.d_address)